import os
import numpy as np
import sounddevice as sd
import soundfile as sf
from resemblyzer import VoiceEncoder, preprocess_wav
from scipy.spatial.distance import cosine
from pathlib import Path

# --- Initialize encoder ---
encoder = VoiceEncoder()

# --- Config ---
DATASET_PATH = "dataset"       # main dataset folder (ahmad, semo, ibrahim)
THRESHOLD = 0.75               # similarity threshold
TEST_FILENAME = "test_voice.wav"

# --- Functions ---
def get_embedding_from_file(file_path):
    """Load or compute embedding for a wav file."""
    embedding_path = file_path.replace(".wav", "_embedding.npy")
    if os.path.exists(embedding_path):
        return np.load(embedding_path)
    else:
        wav = preprocess_wav(Path(file_path))
        emb = encoder.embed_utterance(wav)
        np.save(embedding_path, emb)
        return emb

def record_test_voice(filename, duration=5, samplerate=16000):
    """Record a new voice for testing."""
    print(f"🎙 Recording test voice for {duration} seconds...")
    audio = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype='float32')
    sd.wait()
    sf.write(filename, audio, samplerate)
    print(f"✅ Test voice saved as {filename}")
    return filename

def compare_to_dataset(test_file):
    """Compare the test voice embedding to the dataset."""
    test_emb = get_embedding_from_file(test_file)
    results = {}

    for user_folder in os.listdir(DATASET_PATH):
        folder_path = os.path.join(DATASET_PATH, user_folder, "voice_data")
        if not os.path.exists(folder_path):
            continue

        similarities = []
        for wav_file in os.listdir(folder_path):
            if wav_file.endswith(".wav"):
                wav_path = os.path.join(folder_path, wav_file)
                emb = get_embedding_from_file(wav_path)
                sim = 1 - cosine(test_emb, emb)
                similarities.append(sim)

        if similarities:
            results[user_folder] = max(similarities)

    # Find best match
    if results:
        best_user, best_score = max(results.items(), key=lambda x: x[1])
        print(f"\n🔍 Best match: {best_user} (similarity = {best_score:.2f})")
        if best_score >= THRESHOLD:
            print("✅ Voice recognized! Access granted.")
        else:
            print("❌ Voice not recognized. Access denied.")
    else:
        print("❌ No valid user embeddings found.")

# --- Main ---
if __name__ == "__main__":
    # Step 1: Record a new test voice
    test_file = record_test_voice(TEST_FILENAME)

    # Step 2: Compare to dataset
    compare_to_dataset(test_file)