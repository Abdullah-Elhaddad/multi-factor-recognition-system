import os
import numpy as np
from deepface import DeepFace

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "data", "faces")
EMB_DIR = os.path.join(BASE_DIR, "..", "embeddings")

MODEL_NAME = "Facenet" 
DETECTOR_BACKEND = "opencv"  

os.makedirs(EMB_DIR, exist_ok=True)

def build_user_embedding(user_folder):
    files = sorted([
        os.path.join(user_folder, f)
        for f in os.listdir(user_folder)
        if f.lower().endswith(".jpg")
    ])

    if not files:
        print(f"No image files found in {user_folder}")
        return None, None

    embs = []
    for f in files:
        try:
            result = DeepFace.represent(
                img_path=f,
                model_name=MODEL_NAME,
                enforce_detection=True,
                detector_backend=DETECTOR_BACKEND
            )

            # DeepFace.represent returns a list of dicts
            if isinstance(result, list): 
                result = result[0]["embedding"]

            embs.append(np.array(result))
            print(f"Processed {f}")
        except Exception as e:
            print(f"Error processing {f}: {e}")

    if not embs:
        return None, None

    embs = np.vstack(embs)
    mean_emb = embs.mean(axis=0)
    return mean_emb, embs


def main():
    for username in os.listdir(DATA_DIR):
        user_folder = os.path.join(DATA_DIR, username)
        if not os.path.isdir(user_folder):
            continue

        print(f"\nProcessing user: {username}")
        mean_emb, embs = build_user_embedding(user_folder)

        if mean_emb is None:
            print(f"No embeddings for {username}")
            continue

        np.save(os.path.join(EMB_DIR, f"{username}_mean.npy"), mean_emb)
        np.save(os.path.join(EMB_DIR, f"{username}_all.npy"), embs)
        print(f"Saved embeddings for {username}")


if __name__ == "__main__":
    main()
