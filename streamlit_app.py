import os
import sys
import torch
import librosa
import streamlit as st
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from peft import PeftModel

# Configure Streamlit page header
st.set_page_config(
    page_title="AutoLyrics Demo",
    page_icon="🎵",
    layout="centered"
)

# Custom premium CSS styling for dark mode/purple accent look
st.markdown("""
    <style>
    .main {
        background-color: #fafafa;
    }
    h1 {
        color: #1e1b4b;
        font-weight: 800;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .subtitle {
        text-align: center;
        color: #475569;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    .stButton>button {
        background: linear-gradient(90deg, #4f46e5, #4338ca);
        color: white;
        border: none;
        padding: 0.5rem 2rem;
        border-radius: 0.375rem;
        font-weight: 600;
        width: 100%;
        transition: all 0.2s ease;
    }
    .stButton>button:hover {
        background: linear-gradient(90deg, #6366f1, #4f46e5);
        color: white;
        box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1);
    }
    </style>
""", unsafe_allow_html=True)

# Add src/ to sys path to import helpers
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Global cache to load models once and share them across runs
@st.cache_resource
def load_transcription_engine(model_type, lora_path, base_model="openai/whisper-small"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = WhisperProcessor.from_pretrained(base_model)
    
    if model_type == "Baseline (Zero-Shot)":
        model = WhisperForConditionalGeneration.from_pretrained(base_model)
        active_type = "Baseline (Zero-Shot)"
    else:
        # Check if the adapter configuration file actually exists to avoid empty folder crashes
        config_path = os.path.join(lora_path, "adapter_config.json")
        if not os.path.exists(config_path):
            model = WhisperForConditionalGeneration.from_pretrained(base_model)
            active_type = "Baseline (Zero-Shot) [Fallback]"
        else:
            base = WhisperForConditionalGeneration.from_pretrained(base_model)
            model = PeftModel.from_pretrained(base, lora_path)
            active_type = model_type
            
    model = model.to(device)
    model.eval()
    
    model.generation_config.language = "english"
    model.generation_config.task = "transcribe"
    model.generation_config.forced_decoder_ids = None
    
    return processor, model, device, active_type

# Title section
st.markdown("<h1>🎵 AutoLyrics</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Fine-Tuning Whisper-small for Singing Voice Transcription using Low-Rank Adaptation (LoRA).</p>", unsafe_allow_html=True)

# Quantitative Performance Section
with st.expander("📊 View Quantitative Performance & Experiment Results (English Chunks)"):
    st.markdown("""
        Below are the empirical evaluation metrics achieved across the 3 controlled experiments on the **JamendoLyrics English test split**:
        
        | Configuration | Word Error Rate (WER) | Character Error Rate (CER) | Relative WER Reduction ↓ |
        | :--- | :---: | :---: | :---: |
        | **Experiment 1: Baseline (Zero-Shot)** | 30.40% | 21.20% | — |
        | **Experiment 2: LoRA Decoder Only** | 21.92% | 16.23% | **27.9%** |
        | **Experiment 3: LoRA Encoder + Decoder** | **21.37%** | **15.37%** | **29.7%** |
        
        *Key Takeaway:* Applying LoRA to both the **encoder and decoder** attention layers achieved the peak relative reduction of **29.7%**, proving that adapting acoustic representations (encoder) is essential for handling singer-specific audio variations.
    """)

# File Uploader
audio_file = st.file_uploader("🎤 Upload a Song Clip or Vocal Audio", type=["wav", "mp3", "m4a"])

# Selection dropdown
model_selection = st.selectbox(
    "Select Model Version",
    [
        "Baseline (Zero-Shot)",
        "Experiment 2: LoRA Decoder Only",
        "Experiment 3: LoRA Encoder + Decoder"
    ],
    index=2
)

# Get absolute path of current file's directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Set model checkpoint paths
if model_selection == "Experiment 2: LoRA Decoder Only":
    lora_path = os.path.join(BASE_DIR, "checkpoints", "exp2_decoder", "best")
elif model_selection == "Experiment 3: LoRA Encoder + Decoder":
    lora_path = os.path.join(BASE_DIR, "checkpoints", "exp3_both", "best")
else:
    lora_path = ""

# Transcribe trigger
if st.button("🔮 Transcribe Audio"):
    if audio_file is not None:
        with st.spinner("Processing audio features & running inference..."):
            temp_path = None
            try:
                # 1. Load model via cache
                processor, model, device, active_type = load_transcription_engine(model_selection, lora_path)
                
                # 2. Write buffer to temporary file to support compressed formats like .m4a
                import tempfile
                suffix = os.path.splitext(audio_file.name)[1]
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
                    temp_file.write(audio_file.getvalue())
                    temp_path = temp_file.name
                
                # 3. Resample and extract features
                audio, sr = librosa.load(temp_path, sr=16000)
                inputs = processor(audio, sampling_rate=16000, return_tensors="pt").input_features.to(device)
                
                # 4. Transcribe
                with torch.no_grad():
                    if "Baseline" in active_type:
                        predicted_ids = model.generate(inputs)
                    else:
                        predicted_ids = model.base_model.model.generate(inputs)
                        
                transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
                
                # Show results
                st.success(f"⚡ **Transcription successfully generated using {active_type}!**")
                st.info("📋 **Transcribed Lyrics:**")
                st.markdown(f"> {transcription.strip().lower() if transcription.strip() else '*(No vocal audio transcribed)*'}")
                
            except Exception as e:
                st.error(f"🔴 **Error during processing:** {str(e)}")
            finally:
                # Clean up temporary file
                if temp_path and os.path.exists(temp_path):
                    os.remove(temp_path)
    else:
        st.warning("⚠️ Please upload an audio file first!")

st.markdown("""
    <hr>
    <div style='text-align: center; font-size: 0.8rem; color: #64748b;'>
        Adapted on JamendoLyrics dataset • Peak relative WER reduction: 29.7% • Built with PEFT & Streamlit.
    </div>
""", unsafe_allow_html=True)
