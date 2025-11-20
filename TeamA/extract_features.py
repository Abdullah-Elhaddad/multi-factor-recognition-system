"""
extract_features.py
- Input: username
- Loads images from: dataset/<username>/images/*.jpg  (assumed 160x160 crops)
- Runs FaceNet to get 128-D embeddings
- Saves to: dataset/<username>/extracted_features/embedding_<n>.npy
"""

import os
import numpy as np
from pathlib import Path
from keras_facenet import FaceNet
import cv2

DATASET_ROOT = Path("dataset")

def prewhiten(x: np.ndarray) -> np.ndarray:
    m, s = x.mean(), x.std()
    s = max(s, 1.0 / np.sqrt(x.size))
    return (x - m) / s

def l2_normalize(x: np.ndarray, eps=1e-10) -> np.ndarray:
    return x / np.sqrt(max(np.sum(np.square(x)), eps))

def load_facenet():
    print(f"[OK] Loading FaceNet from keras_facenet library...")
    model = FaceNet()
    return model

def embed_image(model, img_bgr_160):
    # Convert BGR->RGB and use FaceNet's embeddings method
    # FaceNet library handles preprocessing internally
    rgb = cv2.cvtColor(img_bgr_160, cv2.COLOR_BGR2RGB)
    # FaceNet.embeddings() expects shape (num_images, height, width, channels)
    tensor = np.expand_dims(rgb, axis=0)
    emb = model.embeddings(tensor)[0]
    return emb.astype(np.float32)

def main():
    username = input("Username to extract features for: ").strip()
    if not username:
        print("[ERR] Username is required."); return

    user_dir = DATASET_ROOT / username
    img_dir = user_dir / "images"
    out_dir = user_dir / "extracted_features"
    out_dir.mkdir(parents=True, exist_ok=True)

    model = load_facenet()

    images = sorted(img_dir.glob("image.*.jpg"), key=lambda p: p.name)
    if not images:
        print(f"[ERR] No images found in {img_dir}"); return

    for i, img_path in enumerate(images, start=1):
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"[WARN] Could not read {img_path}, skipping."); continue
        if img.shape[:2] != (160, 160):
            # If user replaced images with other sizes, resize safely
            img = cv2.resize(img, (160, 160), interpolation=cv2.INTER_AREA)

        emb = embed_image(model, img)
        out = out_dir / f"embedding_{i}.npy"
        np.save(str(out), emb)
        print(f"[OK] Saved -> {out}")

    print("[DONE] Feature extraction complete.")

if __name__ == "__main__":
    main()
