import os, sys, numpy as np
from src.backend import load_embeddings, cosine_similarity, image_to_embedding  

EMB_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "embeddings"))
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "faces"))

def verify_image(img_path):
    probe = image_to_embedding(img_path)
    if probe is None:
        print("Could not compute embedding for probe")
        return
    emb_map = load_embeddings()
    best_user, best_score = None, -1.0
    for u, e in emb_map.items():
        s = cosine_similarity(probe, e)
        if s > best_score:
            best_score = s; best_user = u
    print("Best:", best_user, best_score)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python src/verify_face.py path/to/image.jpg")
    else:
        verify_image(sys.argv[1])
