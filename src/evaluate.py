import os
import argparse
import re
import torch
from jiwer import wer, cer
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from peft import PeftModel

from preprocess import load_and_split_dataset, chunk_all_songs

def normalize(text):
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", "", text)  # Remove punctuation
    text = text.replace("30", "thirty")
    text = text.replace("&", "and")
    return text

def evaluate(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Evaluating model on device: {device}")
    
    # 1. Load splits and get test set chunks
    _, _, test_songs = load_and_split_dataset(args.language)
    print("Chunking test songs...")
    test_chunks = chunk_all_songs(test_songs)
    
    # 2. Model Loading
    processor = WhisperProcessor.from_pretrained(args.base_model)
    
    if args.lora_dir:
        print(f"Loading base Whisper model: {args.base_model}")
        base_model = WhisperForConditionalGeneration.from_pretrained(args.base_model)
        print(f"Applying fine-tuned LoRA adapters from: {args.lora_dir}")
        model = PeftModel.from_pretrained(base_model, args.lora_dir)
    else:
        print(f"Loading vanilla baseline Whisper model: {args.base_model}")
        model = WhisperForConditionalGeneration.from_pretrained(args.base_model)
        
    model = model.to(device)
    model.eval()
    
    # Configure generation
    model.generation_config.language = "english" if args.language == "en" else args.language
    model.generation_config.task = "transcribe"
    model.generation_config.forced_decoder_ids = None
    
    references = []
    hypotheses = []
    
    print(f"Generating transciptions for {len(test_chunks)} test chunks...")
    print("="*60)
    
    for i, chunk in enumerate(test_chunks):
        inputs = processor(
            chunk['audio_array'],
            sampling_rate=16000,
            return_tensors="pt"
        ).input_features.to(device)
        
        with torch.no_grad():
            if args.lora_dir:
                # PeftModel base_model wrapper
                predicted_ids = model.base_model.model.generate(inputs)
            else:
                predicted_ids = model.generate(inputs)
                
        pred = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        
        raw_ref = chunk['ground_truth'].lower().strip()
        raw_pred = pred.lower().strip()
        
        references.append(raw_ref)
        hypotheses.append(raw_pred)
        
        if i < 5:
            print(f"\n[Test Chunk {i+1}]")
            print(f"  REF : {chunk['ground_truth']}")
            print(f"  PRED: {raw_pred}")
            
    # Normalize before metric calculation
    norm_references = [normalize(r) for r in references]
    norm_hypotheses = [normalize(h) for h in hypotheses]
    
    # Compute WER & CER
    eval_wer = wer(norm_references, norm_hypotheses)
    eval_cer = cer(norm_references, norm_hypotheses)
    
    print("\n" + "="*60)
    print("EVALUATION METRICS RESULTS")
    print("="*60)
    print(f"Word Error Rate (WER)      : {eval_wer:.4f} ({eval_wer * 100:.2f}%)")
    print(f"Character Error Rate (CER) : {eval_cer:.4f} ({eval_cer * 100:.2f}%)")
    
    if args.baseline_wer:
        rel_wer_reduction = (args.baseline_wer - eval_wer) / args.baseline_wer * 100
        print(f"Relative WER Reduction     : {rel_wer_reduction:.2f}% (compared to baseline: {args.baseline_wer * 100:.2f}%)")
    print("="*60)
    
    # Save results to file
    if args.save_results:
        os.makedirs(os.path.dirname(args.save_results), exist_ok=True)
        with open(args.save_results, 'w') as f:
            f.write(f"Model: {args.lora_dir or 'Baseline (Zero-Shot)'}\n")
            f.write(f"WER: {eval_wer:.4f} ({eval_wer * 100:.2f}%)\n")
            f.write(f"CER: {eval_cer:.4f} ({eval_cer * 100:.2f}%)\n")
            if args.baseline_wer:
                f.write(f"Relative WER Reduction: {rel_wer_reduction:.2f}%\n")
        print(f"Results successfully saved to: {args.save_results}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AutoLyrics Evaluation Engine")
    parser.add_argument("--base_model", type=str, default="openai/whisper-small", help="Path or HF ID of base Whisper model")
    parser.add_argument("--lora_dir", type=str, default=None, help="Path to LoRA fine-tuned adapters directory (leave empty for zero-shot baseline)")
    parser.add_argument("--language", type=str, default="en", help="Language code of JamendoLyrics test set")
    parser.add_argument("--baseline_wer", type=float, default=0.3040, help="Baseline zero-shot WER for comparison")
    parser.add_argument("--save_results", type=str, default="results/eval_results.txt", help="Path to write quantitative report")
    
    args = parser.parse_args()
    evaluate(args)
