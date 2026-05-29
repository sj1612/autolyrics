import os
import argparse
import torch
from torch.utils.data import DataLoader
from transformers import WhisperProcessor, WhisperForConditionalGeneration, get_linear_schedule_with_warmup
from peft import get_peft_model, LoraConfig, TaskType

from preprocess import load_and_split_dataset, chunk_all_songs
from dataset import AutoLyricsDataset, collate_lyrics_batch

def get_lora_config(mode="both", rank=8, alpha=32, dropout=0.1):
    target_modules = []
    
    # Decoder layers (both modes use decoder adapters)
    for i in range(12):
        target_modules.extend([
            f"model.decoder.layers.{i}.self_attn.q_proj",
            f"model.decoder.layers.{i}.self_attn.v_proj",
            f"model.decoder.layers.{i}.encoder_attn.q_proj",
            f"model.decoder.layers.{i}.encoder_attn.v_proj",
        ])
        
    # Encoder layers (only added in 'both' mode)
    if mode == "both":
        for i in range(12):
            target_modules.extend([
                f"model.encoder.layers.{i}.self_attn.q_proj",
                f"model.encoder.layers.{i}.self_attn.v_proj",
            ])
            
    return LoraConfig(
        r=rank,
        lora_alpha=alpha,
        target_modules=target_modules,
        lora_dropout=dropout,
        bias="none",
        task_type=TaskType.SEQ_2_SEQ_LM
    )

def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on device: {device}")
    
    # 1. Load splits and preprocess
    train_songs, val_songs, _ = load_and_split_dataset(args.language)
    
    print("Chunking training songs...")
    train_chunks = chunk_all_songs(train_songs)
    print("Chunking validation songs...")
    val_chunks = chunk_all_songs(val_songs)
    
    # 2. Processor and Model Setup
    print("Loading base Whisper model and processor...")
    processor = WhisperProcessor.from_pretrained(args.base_model)
    model = WhisperForConditionalGeneration.from_pretrained(args.base_model)
    
    # Setup generations parameters matching training config
    model.config.use_cache = False
    model.generation_config.language = "english" if args.language == "en" else args.language
    model.generation_config.task = "transcribe"
    model.generation_config.forced_decoder_ids = None
    
    # Apply LoRA
    lora_config = get_lora_config(args.mode, args.lora_rank, args.lora_alpha, args.lora_dropout)
    peft_model = get_peft_model(model, lora_config)
    peft_model = peft_model.to(device)
    peft_model.print_trainable_parameters()
    
    # 3. Create datasets and loaders
    train_dataset = AutoLyricsDataset(train_chunks, processor)
    val_dataset = AutoLyricsDataset(val_chunks, processor)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        collate_fn=collate_lyrics_batch
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        collate_fn=collate_lyrics_batch
    )
    
    # 4. Optimization Setup
    optimizer = torch.optim.AdamW(
        peft_model.parameters(), 
        lr=args.lr, 
        weight_decay=args.weight_decay
    )
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=args.max_steps
    )
    
    scaler = torch.cuda.amp.GradScaler() if device == "cuda" else None
    
    # 5. Training Loop
    step = 0
    best_val_loss = float("inf")
    patience = 0
    
    train_iter = iter(train_loader)
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    print(f"Starting training run (Mode: {args.mode})...")
    print("="*60)
    
    while step < args.max_steps:
        peft_model.train()
        
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)
            
        input_features = batch["input_features"].to(device)
        labels = batch["labels"].to(device)
        
        # Forward pass with mixed precision if GPU
        if device == "cuda":
            with torch.cuda.amp.autocast():
                outputs = peft_model.base_model.model(
                    input_features=input_features,
                    labels=labels
                )
            loss = outputs.loss
            scaler.scale(loss).backward()
            
            # Gradient clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(peft_model.parameters(), 1.0)
            
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = peft_model.base_model.model(
                input_features=input_features,
                labels=labels
            )
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(peft_model.parameters(), 1.0)
            optimizer.step()
            
        scheduler.step()
        optimizer.zero_grad()
        
        step += 1
        
        if step % 10 == 0:
            print(f"Step {step}/{args.max_steps} | Train Loss: {loss.item():.4f}")
            
        # Validation Evaluation
        if step % args.eval_every == 0:
            peft_model.eval()
            val_loss_total = 0
            val_steps = 0
            
            with torch.no_grad():
                for val_batch in val_loader:
                    val_features = val_batch["input_features"].to(device)
                    val_labels = val_batch["labels"].to(device)
                    
                    if device == "cuda":
                        with torch.cuda.amp.autocast():
                            val_outputs = peft_model.base_model.model(
                                input_features=val_features,
                                labels=val_labels
                            )
                    else:
                        val_outputs = peft_model.base_model.model(
                            input_features=val_features,
                            labels=val_labels
                        )
                    val_loss_total += val_outputs.loss.item()
                    val_steps += 1
                    
            avg_val_loss = val_loss_total / val_steps
            print(f"  → Val Loss at step {step}: {avg_val_loss:.4f}")
            
            # Checkpoint and Early Stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience = 0
                checkpoint_path = os.path.join(args.save_dir, "best")
                peft_model.save_pretrained(checkpoint_path)
                processor.save_pretrained(checkpoint_path)
                print(f"  → New best checkpoint saved to: {checkpoint_path} (val loss: {best_val_loss:.4f})")
            else:
                patience += 1
                print(f"  → No improvement. Patience: {patience}/{args.patience}")
                if patience >= args.patience:
                    print("  → Early stopping triggered. Ending training run.")
                    break
                    
    print("\n✓ Training finished successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AutoLyrics Whisper Fine-Tuning")
    parser.add_argument("--base_model", type=str, default="openai/whisper-small", help="Path or HF ID of base Whisper model")
    parser.add_argument("--mode", type=str, default="both", choices=["decoder", "both"], help="Where to inject LoRA parameters")
    parser.add_argument("--language", type=str, default="en", help="Language code of JamendoLyrics to train on")
    parser.add_argument("--save_dir", type=str, default="checkpoints/exp3_both", help="Checkpoint directory")
    
    # Hyperparams
    parser.add_argument("--max_steps", type=int, default=100, help="Number of training steps")
    parser.add_argument("--eval_every", type=int, default=20, help="Steps between validation evaluations")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size per GPU")
    parser.add_argument("--lr", type=float, default=3e-5, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="AdamW weight decay")
    parser.add_argument("--warmup_steps", type=int, default=10, help="Scheduler warmup steps")
    parser.add_argument("--patience", type=int, default=3, help="Early stopping patience")
    
    # LoRA config
    parser.add_argument("--lora_rank", type=int, default=8, help="LoRA rank (r)")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA scaling alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.1, help="LoRA dropout rate")
    
    args = parser.parse_args()
    train(args)
