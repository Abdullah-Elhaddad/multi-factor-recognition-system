"""
Multifactor Voice Data Collection System + Noise Filtering + Low-Pass Filter
----------------------------------------------------------------------------
This script records multiple voice samples, applies a noise reduction filter
and a low-pass filter using NumPy, and saves clean .wav files for AI model training.
"""

import os
import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write


# -------------------------------------------------
# Configuration
# -------------------------------------------------
SENTENCES = [
    "My voice is my password, verify me.",
    "Security is important for all systems.",
    "Artificial intelligence can recognize voices.",
    "Voice and face make a strong protection system.",
    "This is the final sentence for verification."
]

SAMPLE_RATE = 44100  # Hz
DURATION = 5          # seconds


# -------------------------------------------------
# Low-Pass Filter using FFT
# -------------------------------------------------
def low_pass_filter(audio_data: np.ndarray, cutoff_freq: float = 8000.0) -> np.ndarray:
    """
    Apply a low-pass filter using NumPy FFT with smooth rolloff.
    Removes high-frequency noise while preserving voice clarity.
    Uses 8000 Hz cutoff to maintain natural voice quality.
    """
    # Convert to float for processing
    audio_float = audio_data.astype(np.float32).flatten()
    
    # Perform Fast Fourier Transform
    fft_data = np.fft.rfft(audio_float)
    freqs = np.fft.rfftfreq(len(audio_float), d=1.0 / SAMPLE_RATE)

    # Create a smooth frequency mask (smooth rolloff instead of hard cutoff)
    # This prevents artifacts and preserves voice clarity
    transition_width = 1000.0  # Hz transition band
    transition_start = cutoff_freq - transition_width / 2
    transition_end = cutoff_freq + transition_width / 2
    
    # Initialize mask: keep all frequencies below transition_start
    mask = np.ones_like(freqs)
    
    # Apply smooth rolloff in transition band (vectorized)
    transition_mask = (freqs > transition_start) & (freqs <= transition_end)
    if np.any(transition_mask):
        t = (freqs[transition_mask] - transition_start) / transition_width
        mask[transition_mask] = 0.5 * (1 + np.cos(np.pi * t))
    
    # Remove frequencies above transition_end
    mask[freqs > transition_end] = 0.0
    
    fft_data_filtered = fft_data * mask

    # Perform inverse FFT to return to time domain
    filtered_audio = np.fft.irfft(fft_data_filtered)

    # Normalize
    filtered_audio /= np.max(np.abs(filtered_audio)) + 1e-6
    filtered_audio = (filtered_audio * 32767).astype(np.int16)

    return filtered_audio.reshape(audio_data.shape)


# -------------------------------------------------
# Noise Reduction (Basic Filter)
# -------------------------------------------------
def noise_reduction(audio_data: np.ndarray) -> np.ndarray:
    """
    Simple noise reduction: normalize, smooth, threshold.
    """
    audio_float = audio_data.astype(np.float32)
    audio_float /= np.max(np.abs(audio_float))

    # Moving average smoothing
    window_size = 5
    kernel = np.ones(window_size) / window_size
    smoothed = np.convolve(audio_float.flatten(), kernel, mode='same')

    # Remove low-level noise
    threshold = 0.01
    smoothed[np.abs(smoothed) < threshold] = 0

    smoothed /= np.max(np.abs(smoothed)) + 1e-6
    clean_audio = (smoothed * 32767).astype(np.int16)

    return clean_audio.reshape(audio_data.shape)


# -------------------------------------------------
# Filename Handling
# -------------------------------------------------
def get_next_filename(folder_path: str, base_name: str) -> str:
    """
    Auto-increment file name if it already exists.
    """
    base_path = os.path.join(folder_path, base_name)
    if not os.path.exists(base_path):
        return base_name

    name_no_ext = base_name.rsplit('.', 1)[0]
    ext = base_name.rsplit('.', 1)[1] if '.' in base_name else ''
    counter = 1
    while True:
        new_name = f"{name_no_ext}_{counter}.{ext}" if ext else f"{name_no_ext}_{counter}"
        if not os.path.exists(os.path.join(folder_path, new_name)):
            return new_name
        counter += 1


# -------------------------------------------------
# Recording Function
# -------------------------------------------------
def record_sentence(username: str, sentence: str, index: int):
    """Record, clean, apply low-pass filter, and save."""
    print(f"\nSentence {index + 1}/{len(SENTENCES)}: {sentence}")
    input("Press ENTER when you're ready to record...")

    print("🎙 Recording... Speak clearly now.")
    recording = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='int16')
    sd.wait()
    print("✅ Recording finished.")

    # Apply noise reduction
    print("🧹 Cleaning background noise...")
    cleaned = noise_reduction(recording)

    # Apply low-pass filter (8000 Hz preserves voice clarity while removing high-frequency noise)
    print("🔉 Applying low-pass filter for voice clarity...")
    filtered = low_pass_filter(cleaned, cutoff_freq=8000.0)

    # Save clean audio
    folder_path = os.path.join("dataset", username, "voice_data")
    os.makedirs(folder_path, exist_ok=True)

    base_file_name = f"{username}_sentence{index + 1}.wav"
    file_name = get_next_filename(folder_path, base_file_name)
    file_path = os.path.join(folder_path, file_name)
    write(file_path, SAMPLE_RATE, filtered)

    print(f"💾 Clean filtered voice saved as: {file_path}")


# -------------------------------------------------
# Main Program
# -------------------------------------------------
def main():
    print("=======================================")
    print(" MULTIFACTOR VOICE DATA COLLECTION")
    print("=======================================")

    username = input("\nEnter your username: ").strip()
    if not username:
        print("⚠ Username cannot be empty. Exiting...")
        return

    print(f"\nWelcome, {username}!")
    print("You will record 5 sentences, each lasting 5 seconds.\n")

    for idx, sentence in enumerate(SENTENCES):
        record_sentence(username, sentence, idx)

    print("\n✅ All clean recordings completed successfully!")
    print(f"Voice data saved under: dataset/{username}/voice_data/")


# -------------------------------------------------
# Run the program
# -------------------------------------------------
if __name__ == "__main__":
    main()