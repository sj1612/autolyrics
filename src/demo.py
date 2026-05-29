import os
import argparse
import torch
import gradio as gr
import librosa
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from peft import PeftModel

# Global cache to prevent re-loading same models repeatedly
MODEL_CACHE = {}

def get_transcription_engine(model_type, lora_path, base_model="openai/whisper-small"):
    cache_key = (model_type, lora_path)
    if cache_key in MODEL_CACHE:
        return MODEL_CACHE[cache_key]
        
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {model_type} model on {device}...")
    
    processor = WhisperProcessor.from_pretrained(base_model)
    
    if model_type == "Baseline (Zero-Shot)":
        model = WhisperForConditionalGeneration.from_pretrained(base_model)
    else:
        # Check if the adapter configuration file actually exists to avoid empty folder crashes
        config_path = os.path.join(lora_path, "adapter_config.json")
        if not os.path.exists(config_path):
            print(f"Warning: Checkpoint config at {config_path} not found. Falling back to vanilla baseline.")
            model = WhisperForConditionalGeneration.from_pretrained(base_model)
            model_type = "Baseline (Zero-Shot) [Fallback]"
        else:
            base = WhisperForConditionalGeneration.from_pretrained(base_model)
            model = PeftModel.from_pretrained(base, lora_path)
            
    model = model.to(device)
    model.eval()
    
    # Configure generation
    model.generation_config.language = "english"
    model.generation_config.task = "transcribe"
    model.generation_config.forced_decoder_ids = None
    
    MODEL_CACHE[cache_key] = (processor, model, device, model_type)
    return MODEL_CACHE[cache_key]

def transcribe_audio(audio_file, model_selection):
    if audio_file is None:
        return "⚠️ Please record or upload an audio file first."
        
    # Get absolute path of project root directory (parent of src/)
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Get parameters matching selection
    if model_selection == "Experiment 2: LoRA Decoder Only":
        lora_path = os.path.join(BASE_DIR, "checkpoints", "exp2_decoder", "best")
    elif model_selection == "Experiment 3: LoRA Encoder + Decoder":
        lora_path = os.path.join(BASE_DIR, "checkpoints", "exp3_both", "best")
    else:
        lora_path = ""
        
    try:
        processor, model, device, active_type = get_transcription_engine(model_selection, lora_path)
        
        # Load audio and resample to 16kHz
        audio, sr = librosa.load(audio_file, sr=16000)
        
        # Extract features
        inputs = processor(audio, sampling_rate=16000, return_tensors="pt").input_features.to(device)
        
        # Transcribe
        with torch.no_grad():
            if "Baseline" in active_type:
                predicted_ids = model.generate(inputs)
            else:
                predicted_ids = model.base_model.model.generate(inputs)
                
        transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        
        clean_text = transcription.strip().lower()
        if not clean_text:
            return "*(No vocal audio transcribed)*"
            
        status = f"⚡ **Transcription successfully generated using {active_type}!**\n\n"
        return status + f"***\n\n📋 **Transcribed Lyrics:**\n\n> {clean_text}\n\n***"
        
    except Exception as e:
        return f"🔴 **Error during processing:** {str(e)}"

# Custom premium theme & layout styling
custom_theme = gr.themes.Soft(
    primary_hue="indigo",
    secondary_hue="slate",
    neutral_hue="zinc",
    font=[gr.themes.GoogleFont("Outfit"), "sans-serif"]
).set(
    body_background_fill="*neutral_50",
    block_background_fill="white",
    block_border_width="1px",
    block_shadow="0 4px 6px -1px rgb(0 0 0 / 0.1)",
    button_primary_background_fill="linear-gradient(90deg, *primary_600, *primary_700)",
    button_primary_background_fill_hover="linear-gradient(90deg, *primary_500, *primary_600)"
)

# Build the layout
with gr.Blocks(theme=custom_theme, title="AutoLyrics Demo") as demo:
    gr.Markdown(
        """
        <div style="text-align: center; margin-bottom: 2rem;">
            <span style="font-size: 3rem;">🎵</span>
            <h1 style="font-size: 2.5rem; font-weight: 800; color: #1e1b4b; margin: 0.5rem 0;">AutoLyrics</h1>
            <p style="font-size: 1.1rem; color: #475569; max-width: 600px; margin: 0 auto;">
                Fine-Tuning Whisper-small for Singing Voice Transcription using Low-Rank Adaptation (LoRA). Upload a clip or record yourself singing!
            </p>
        </div>
        """
    )
    
    gr.Markdown(
        """
        ### 📊 Controlled Experiment Results (English Chunks)
        
        | Configuration | Word Error Rate (WER) | Character Error Rate (CER) | Relative WER Reduction ↓ |
        | :--- | :---: | :---: | :---: |
        | **Experiment 1: Baseline (Zero-Shot)** | 30.40% | 21.20% | — |
        | **Experiment 2: LoRA Decoder Only** | 21.92% | 16.23% | **27.9%** |
        | **Experiment 3: LoRA Encoder + Decoder** | **21.37%** | **15.37%** | **29.7%** |
        
        *Key Takeaway:* Fine-tuning both the **encoder and decoder** attention layers achieved the peak relative reduction of **29.7%**, proving that adapting acoustic representations is crucial for handling singing voice variations.
        
        ---
        """
    )
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 🎤 Audio Input & Model Setup")
            
            audio_input = gr.Audio(
                sources=["upload", "microphone"],
                type="filepath",
                label="Audio Track"
            )
            
            model_selection = gr.Dropdown(
                choices=[
                    "Baseline (Zero-Shot)",
                    "Experiment 2: LoRA Decoder Only",
                    "Experiment 3: LoRA Encoder + Decoder"
                ],
                value="Experiment 3: LoRA Encoder + Decoder",
                label="Select Model Version"
            )
            
            transcribe_btn = gr.Button("🔮 Transcribe Audio", variant="primary")
            
        with gr.Column(scale=1):
            gr.Markdown("### 📝 Output Lyrics")
            output_display = gr.Markdown(
                "💡 *Upload/record an audio clip and click Transcribe to view the results here.*",
                elem_id="output-markdown"
            )
            
    # Connect trigger
    transcribe_btn.click(
        fn=transcribe_audio,
        inputs=[audio_input, model_selection],
        outputs=[output_display]
    )
    
    gr.Markdown(
        """
        ---
        <div style="text-align: center; font-size: 0.85rem; color: #64748b;">
            Adapted on JamendoLyrics dataset • Peak relative WER reduction: 29.7% • Built with PEFT & Gradio.
        </div>
        """
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AutoLyrics Interactive Demo")
    parser.add_argument("--port", type=int, default=7860, help="Local host port to serve app")
    parser.add_argument("--share", action="store_true", help="Generate public HF/gradio URL")
    args = parser.parse_args()
    
    demo.launch(server_port=args.port, share=args.share)
