import torch
from torch.utils.data import Dataset

class AutoLyricsDataset(Dataset):
    """
    Custom PyTorch Dataset that uses a WhisperProcessor
    to extract log-Mel spectrograms from audio and tokenize ground-truth lyrics.
    """
    def __init__(self, chunks, processor):
        self.chunks = chunks
        self.processor = processor

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        chunk = self.chunks[idx]
        
        # Extract features (Log-Mel Spectrogram)
        inputs = self.processor(
            chunk['audio_array'],
            sampling_rate=16000,
            return_tensors="pt"
        )
        
        # Tokenize labels (lyrics text)
        labels = self.processor.tokenizer(
            chunk['ground_truth'],
            return_tensors="pt"
        ).input_ids

        return {
            "input_features": inputs.input_features.squeeze(0),  # Shape: (80, 3000)
            "labels": labels.squeeze(0)                          # Shape: (seq_len,)
        }

def collate_lyrics_batch(batch):
    """
    Dynamic padding collator for Whisper training batches.
    Pads targets with -100 to ignore them in standard cross-entropy loss computation.
    """
    input_features = torch.stack([b["input_features"] for b in batch])
    labels = [b["labels"] for b in batch]
    
    # Pad label sequences to the maximum length in the current batch
    max_len = max(l.shape[0] for l in labels)
    padded_labels = torch.full((len(labels), max_len), -100, dtype=torch.long)
    for i, l in enumerate(labels):
        padded_labels[i, :l.shape[0]] = l
        
    return {
        "input_features": input_features,
        "labels": padded_labels
    }
