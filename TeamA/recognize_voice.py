"""
Voice Verification System
-------------------------
This script records a new voice sample, extracts MFCC features,
compares it with stored dataset samples, and decides if the voice
matches the user (Access Granted / Denied).
"""

import os
import numpy as np
import sounddevice as sd
import librosa
from scipy.io.wavfile import write
from sklearn.metrics.pairwise import cosine_similarity

# -------------------------------------------------
# Configuration
# -------------------------------------------------
SAMPLE_RATE = 44100
DURATION = 5
THRESHOLD = 0.80  # similarity threshold (tune if needed)

# -------------------------------------------------
# Helper Functions
# -------------------------------------------------
def record_voice(temp_file="temp_voice.wav"):
    """Record a new voice sample."""
    print("\n🎙 Speak now to verify your voice...")
    recording = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='int16')
    sd.wait()
    write(temp_file, SAMPLE_RATE, recording)
    print("✅ Voice sample recorded and saved.")
    return temp_file


def extract_features(file_path):
    """Extract MFCC features from an audio file."""
    y, sr = librosa.load(file_path, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    mfcc_mean = np.mean(mfcc.T, axis=0)  # average across time
    return mfcc_mean


def verify_user(username):
    """Verify if the new voice matches the user's dataset."""
    user_folder = os.path.join("dataset", username, "voice_data")
    if not os.path.exists(user_folder):
        print(f"⚠ No voice data found for user '{username}'. Please record first.")
        return

    # Record new voice
    new_voice_path = record_voice()

    # Extract features for new voice
    new_features = extract_features(new_voice_path)

    similarities = []

    # Compare with all existing samples
    for file in os.listdir(user_folder):
        if file.endswith(".wav"):
            file_path = os.path.join(user_folder, file)
            dataset_features = extract_features(file_path)
            sim = cosine_similarity([new_features], [dataset_features])[0][0]
            similarities.append(sim)

    avg_similarity = np.mean(similarities)
    print(f"\n🔍 Average voice similarity: {avg_similarity:.2f}")

    if avg_similarity >= THRESHOLD:
        print("✅ Access Granted — Voice Match Confirmed!")
    else:
        print("❌ Access Denied — Voice Mismatch.")


# -------------------------------------------------
# Main Program
# -------------------------------------------------
def main():
    print("=======================================")
    print("     MULTIFACTOR VOICE VERIFICATION")
    print("=======================================")

    username = input("\nEnter username to verify: ").strip()
    if not username:
        print("⚠ Username cannot be empty.")
        return

    verify_user(username)


# -------------------------------------------------
# Run
# -------------------------------------------------
if __name__ == "__main__":
    main()