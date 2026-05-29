Live demo- https://autolyrics-b8hbwps8sxnrsprjl2gzdk.streamlit.app/
# AutoLyrics 
### Fine-Tuning Whisper for Singing Voice Transcription using LoRA

> Fine-tuned OpenAI Whisper-small for singing voice transcription using parameter-efficient LoRA adaptation, achieving **29.7% relative WER reduction** over the zero-shot baseline across 3 controlled experiments on the JamendoLyrics dataset.

---

## Overview

Standard ASR systems are trained on spoken language and struggle significantly with singing audio due to pitch variations, prolonged phonemes, rhythmic pacing, and overlapping background instrumentation. **AutoLyrics** addresses this by adapting Whisper-small for lyric transcription using LoRA (Low-Rank Adaptation) — a parameter-efficient fine-tuning technique that trains only ~1-2% of model parameters while keeping the rest frozen.

---

## Results
=======
# AutoLyrics 🎵
### Fine-Tuning Whisper for Singing Voice Transcription using LoRA

> Adapted OpenAI's Whisper-small for singing voice transcription using parameter-efficient LoRA adaptation, achieving a **29.7% relative WER reduction** over the zero-shot baseline across controlled experiments on the JamendoLyrics dataset. Now equipped with a production-grade modular pipeline and deployable web demos!

---

## 🚀 Key Achievements & Results
>>>>>>> 261f71a (Initial commit: Modular code and deployable web apps)

| Experiment | WER | CER | Relative WER Reduction |
|---|---|---|---|
| Baseline (Zero-shot Whisper) | 30.40% | 21.20% | — |
| Exp 2: LoRA Decoder Only | 21.92% | 16.23% | **27.9%** |
| Exp 3: LoRA Encoder + Decoder | 21.37% | 15.37% | **29.7%** |

<<<<<<< HEAD
Key finding: Applying LoRA to both encoder and decoder outperforms decoder-only adaptation, demonstrating that acoustic feature adaptation (encoder) is important for singing-specific inputs, not just text generation (decoder).

---

## Architecture

```
Singing Audio (.wav)
        ↓
  Resample to 16kHz
        ↓
  Log-Mel Spectrogram (WhisperProcessor)
        ↓
  Whisper Encoder (LoRA adapted)
        ↓
  Whisper Decoder (LoRA adapted)
        ↓
  Predicted Lyrics (text)
        ↓
  WER / CER Evaluation (jiwer)
=======
*Key finding: Applying LoRA to both the encoder and decoder outperforms decoder-only adaptation, demonstrating that acoustic feature adaptation (encoder) is vital for singing-specific inputs (pitch shifts, vibratos), not just language generation (decoder).*

---

## 📁 Project Architecture

The project has been refactored from a monolithic notebook into a production-grade modular codebase:

```
autolyrics/
├── src/
│   ├── preprocess.py      ← resampling (16kHz), song-level partitioning, and 30s chunking
│   ├── dataset.py         ← PyTorch custom dataset and dynamic padding collator
│   ├── train.py           ← fully parameterized early-stopping trainer (supports multiple languages)
│   ├── evaluate.py        ← JIWER evaluation engine with text normalizations
│   └── demo.py            ← Gradio web app with premium aesthetic dark-mode UI
├── app.py                 ← HF Spaces deployment entry point
├── streamlit_app.py       ← Streamlit Community Cloud web app
├── requirements.txt       ← Pip dependency list
└── README.md              ← You are here!
>>>>>>> 261f71a (Initial commit: Modular code and deployable web apps)
```

---

<<<<<<< HEAD
## Dataset

- **Source:** [JamendoLyrics](https://huggingface.co/datasets/jamendolyrics/jamendolyrics) — royalty-free music with word-level time-aligned lyrics
- **Subset:** 20 English songs filtered from 79 multilingual songs
- **Chunking:** Songs split into 30-second segments → 94 train / 21 val / 22 test chunks
- **Split strategy:** Split by song (not randomly) to prevent data leakage

---

## LoRA Configuration

| Parameter | Value |
|---|---|
| Rank (r) | 8 |
| Alpha | 32 |
| Dropout | 0.1 |
| Target modules | q_proj, v_proj (attention layers) |
| Trainable parameters | ~1-2% of total |
| Task type | Seq2Seq LM |

---

## Experiments

**Experiment 1 — Zero-shot Baseline**
Vanilla Whisper-small evaluated on singing audio with no modifications. Establishes reference WER/CER.

**Experiment 2 — LoRA Decoder Only**
LoRA adapters injected into decoder self-attention and cross-attention layers. Improves lyric generation quality.

**Experiment 3 — LoRA Encoder + Decoder**
LoRA applied to both encoder and decoder attention layers. Best performance — encoder adaptation helps the model learn singing-specific acoustic patterns like pitch variations and vibrato.

---

## Tech Stack

| Category | Tools |
|---|---|
| Base Model | OpenAI Whisper-small |
| Fine-tuning | PEFT / LoRA (HuggingFace) |
| Framework | PyTorch, HuggingFace Transformers |
| Dataset | HuggingFace Datasets |
| Audio | Torchaudio |
| Evaluation | jiwer (WER / CER) |
| Training | Google Colab (T4 GPU) |

---


---

## Setup

```bash
git clone https://github.com/sj1612/autolyrics
cd autolyrics
pip install -r requirements.txt
```

**requirements.txt**
```
transformers>=4.40.0
datasets>=2.21.0
peft>=0.10.0
accelerate>=1.1.0
torchaudio
jiwer
gradio
torch
=======
## 🛠️ Installation & Setup

1. Clone the repository and navigate into the folder:
```bash
git clone https://github.com/<YOUR-USERNAME>/autolyrics.git
cd autolyrics
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

---

## 🏃 Running the Pipeline

All scripts support a robust Command Line Interface (CLI) allowing you to customize languages, checkpoints, and parameters.

### 1. Data Preprocessing
Preprocess and inspect chunk splits for any target language (e.g., English `en`, French `fr`, Spanish `es`):
```bash
python src/preprocess.py --language en
```

### 2. Fine-Tuning (LoRA)
Fine-tune Whisper-small on singing audio using early-stopping custom PyTorch loops:
```bash
# Fine-tune on English songs
python src/train.py --language en --mode both --save_dir checkpoints/exp3_both_en

# Fine-tune on French songs
python src/train.py --language fr --mode both --save_dir checkpoints/exp3_both_fr
```

### 3. Quantitative Evaluation
Evaluate any trained adapter against the baseline zero-shot model:
```bash
# Evaluate English baseline
python src/evaluate.py --language en

# Evaluate your custom English LoRA adapter
python src/evaluate.py --language en --lora_dir checkpoints/exp3_both_en/best --save_results results/lora_en_results.txt
```

### 4. Interactive Web Demos
Run either user interface locally on your machine:
```bash
# Launch Gradio Demo
python src/demo.py --port 7860

# Launch Streamlit Demo
streamlit run streamlit_app.py
>>>>>>> 261f71a (Initial commit: Modular code and deployable web apps)
```

---

<<<<<<< HEAD
## Qualitative Analysis

Example output comparison on a held-out test song:

| | Text |
|---|---|
| **Ground Truth** | send thirty messages drunk and dirty messages drunk but i don't look no i can't be to look i could not breathe you took your toll on me |
| **Baseline Whisper** | send 30 messages drunk and dirty messages drunk but i don't look no i can't be put to luck i cannot breathe you took your toll on me |
| **AutoLyrics (Exp 3)** | send thirty messages drunk and dirty messages drunk but i don't look no i can't be to look i could not breathe you took your toll on me |

Notable improvements after fine-tuning:
- Number normalization ("30" → "thirty")
- Reduced hallucination on sustained notes
- Better handling of repeated chorus sections
- Improved phoneme disambiguation under instrumental accompaniment

---

## Challenges

- **Small dataset (94 chunks):** Required careful regularization (weight decay, early stopping) to prevent overfitting
- **Polyphonic audio:** Instrumental accompaniment masks vocal phonemes — a fundamental challenge for singing ASR
- **Chunking alignment:** Word-level timestamps from JamendoLyrics used to pair each 30s audio chunk with correct ground truth lyrics
- **PEFT + Whisper compatibility:** Required custom training loop to bypass a PEFT routing bug with Whisper's encoder-decoder architecture

---

## Future Work

- Train on larger singing datasets (DALI, MIR-1K)
- Apply vocal source separation before transcription
- Experiment with larger Whisper variants (medium, large-v3)
- Add timestamp prediction for karaoke-style alignment
- Deploy as real-time Gradio web application

---

## References

- Radford et al., [Robust Speech Recognition via Large-Scale Weak Supervision](https://arxiv.org/abs/2212.04356) (Whisper)
- Hu et al., [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- Durand et al., [Contrastive Learning-Based Audio to Lyrics Alignment](https://arxiv.org/abs/2306.07744) (JamendoLyrics)
- [HuggingFace PEFT Documentation](https://huggingface.co/docs/peft)
- [JiWER — ASR Evaluation Library](https://github.com/jitsi/jiwer)

---

*Built by Sanjana Jayaganthan — IIT Guwahati (240102083)*
=======
## ☁️ Deployment

This project is configured out-of-the-box for **1-click free cloud hosting**:

*   **Hugging Face Spaces**: Integrates directly with the root-level `app.py`. Just create a Gradio Space on Hugging Face and upload the files.
*   **Streamlit Community Cloud**: Integrates directly with `streamlit_app.py`. Connect your public GitHub repository to Streamlit Cloud to launch!

---

## 🔬 Qualitative Analysis Examples

Notable improvements after fine-tuning include:
*   **Number Normalization**: Caster text normalization maps numeric values correctly (`"30"` $\rightarrow$ `"thirty"`).
*   **Reduced Hallucinations**: Standard Whisper often hallucinates or gets stuck in a loop on sustained vocal notes; the adapted model manages sustained vocals and overlapping instrumentation smoothly.

---

## 🔮 Future Work
*   Train on massive multi-lingual datasets (DALI, MIR-1K).
*   Apply vocal source separation (e.g., Demucs) as a preprocessing pipeline step to isolate vocal tracks before transcribing.
*   Integrate real-time vocal timestamping for karaoke-style visual outputs.

---

*Project developed and engineered by Sanjana Jayaganthan — IIT Guwahati (240102083)*
>>>>>>> 261f71a (Initial commit: Modular code and deployable web apps)
