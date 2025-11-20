# build_profile.py
# Load dataset/<username>/extracted_features/*.npy
# Save L2-normalized mean vector to dataset/<username>/profile.npz

import numpy as np
from pathlib import Path

DATASET_ROOT = Path("dataset")

def l2_normalize(x, eps=1e-10):
    return x / max(np.linalg.norm(x), eps)

def main():
    username = input("Username to build profile for: ").strip()
    if not username:
        print("[ERR] Username is required."); return

    feat_dir = DATASET_ROOT / username / "extracted_features"
    feats = sorted(feat_dir.glob("embedding_*.npy"))
    if not feats:
        print(f"[ERR] No embeddings in {feat_dir}"); return

    arr = [np.load(str(p)).astype(np.float32) for p in feats]
    mean_vec = l2_normalize(np.mean(np.stack(arr, axis=0), axis=0).astype(np.float32))
    out_path = DATASET_ROOT / username / "profile.npz"
    np.savez_compressed(out_path, embedding_mean=mean_vec, count=len(arr))
    print(f"[OK] Saved profile -> {out_path}")

if __name__ == "__main__":
    main()
