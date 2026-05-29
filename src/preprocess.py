import os
import argparse
import numpy as np
from datasets import load_dataset, Audio

SAMPLE_RATE = 16000
CHUNK_DURATION = 30  # seconds

def chunk_song(song, chunk_duration=CHUNK_DURATION, sample_rate=SAMPLE_RATE):
    """
    Splits one song into 30-second audio chunks,
    each paired with its ground truth lyrics.
    """
    audio_array = song['audio']['array']
    words = song['words']
    total_duration = len(audio_array) / sample_rate

    chunks = []
    chunk_start_time = 0.0

    while chunk_start_time < total_duration:
        chunk_end_time = chunk_start_time + chunk_duration

        start_sample = int(chunk_start_time * sample_rate)
        end_sample = min(int(chunk_end_time * sample_rate), len(audio_array))
        audio_chunk = audio_array[start_sample:end_sample]

        if len(audio_chunk) < 5 * sample_rate:
            chunk_start_time = chunk_end_time
            continue

        chunk_words = [
            w['text'] for w in words
            if w['start'] >= chunk_start_time and w['start'] < chunk_end_time
        ]

        if not chunk_words:
            chunk_start_time = chunk_end_time
            continue

        ground_truth = ' '.join(chunk_words).strip().lower()

        chunks.append({
            'audio_array': audio_chunk,
            'ground_truth': ground_truth,
            'song_title': song['title'],
            'chunk_start': chunk_start_time,
            'chunk_end': chunk_end_time,
        })

        chunk_start_time = chunk_end_time

    return chunks

def chunk_all_songs(song_split, chunk_duration=CHUNK_DURATION, sample_rate=SAMPLE_RATE):
    all_chunks = []
    for song in song_split:
        chunks = chunk_song(song, chunk_duration, sample_rate)
        all_chunks.extend(chunks)
        print(f"  {song['title']}: {len(chunks)} chunks")
    return all_chunks

def load_and_split_dataset(language="en", train_indices=None, val_indices=None, test_indices=None):
    print(f"Loading JamendoLyrics dataset (language: {language})...")
    # JamendoLyrics test split has all the songs
    dataset = load_dataset("jamendolyrics/jamendolyrics", language, split="test")
    dataset = dataset.cast_column("audio", Audio(sampling_rate=SAMPLE_RATE))

    total_songs = len(dataset)
    print(f"Total songs loaded: {total_songs}")

    # Set default splits if not provided
    if train_indices is None or val_indices is None or test_indices is None:
        if language == "en":
            train_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
            val_indices = [14, 15, 16]
            test_indices = [17, 18, 19]
        else:
            # Simple automatic split based on total count
            indices = list(range(total_songs))
            train_end = int(total_songs * 0.7)
            val_end = int(total_songs * 0.85)
            train_indices = indices[:train_end]
            val_indices = indices[train_end:val_end]
            test_indices = indices[val_end:]

    print(f"Split indices - Train: {train_indices}, Val: {val_indices}, Test: {test_indices}")
    
    train_songs = dataset.select(train_indices)
    val_songs = dataset.select(val_indices)
    test_songs = dataset.select(test_indices)

    return train_songs, val_songs, test_songs

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AutoLyrics JamendoLyrics Preprocessing")
    parser.add_argument("--language", type=str, default="en", help="Dataset language (e.g., en, fr, de, es)")
    parser.add_argument("--dry-run", action="store_true", help="Perform a dry run without saving files")
    args = parser.parse_args()

    train_songs, val_songs, test_songs = load_and_split_dataset(args.language)
    
    print("Chunking training songs...")
    train_chunks = chunk_all_songs(train_songs)
    print("Chunking validation songs...")
    val_chunks = chunk_all_songs(val_songs)
    print("Chunking test songs...")
    test_chunks = chunk_all_songs(test_songs)

    print(f"\nPreprocessing Complete!")
    print(f"Train chunks: {len(train_chunks)}")
    print(f"Val chunks: {len(val_chunks)}")
    print(f"Test chunks: {len(test_chunks)}")
