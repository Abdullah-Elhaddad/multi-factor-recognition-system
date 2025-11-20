"""
test_application.py
Usage:
    python test_application.py

- Loads FaceNet model
- Loads all user prototypes from dataset/*/profile.npz
- Captures image from camera
- Detects largest face in the captured image, embeds it
- Computes cosine similarity to each profile -> best match + match %
- If similarity < TAU, returns Unknown
"""

import os
import numpy as np
import cv2
from pathlib import Path
from keras_facenet import FaceNet

# ---------------- Config ----------------
DATASET_ROOT = Path("dataset")
IMG_SIZE     = (160, 160)
TAU          = 0.80   # similarity threshold for "Known" (tune on your data)
CAMERA_ID    = 0      # default camera (0 for built-in, 1+ for external)

# --------------- Utils ------------------
def prewhiten(x: np.ndarray) -> np.ndarray:
    m, s = x.mean(), x.std()
    s = max(s, 1.0 / np.sqrt(x.size))
    return (x - m) / s

def l2_normalize(x: np.ndarray, eps=1e-10) -> np.ndarray:
    n = np.linalg.norm(x)
    return x / max(n, eps)

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    # If a and b are L2-normalized, dot == cosine
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))

def detect_largest_face_bgr(img_bgr: np.ndarray):
    """Return (x,y,w,h) of largest frontal face or None."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    if cascade.empty():
        raise RuntimeError("Could not load Haar cascade.")
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5,
                                     minSize=(60, 60), flags=cv2.CASCADE_SCALE_IMAGE)
    if len(faces) == 0:
        return None
    return tuple(max(faces, key=lambda r: r[2]*r[3]))  # (x,y,w,h)

def crop_align_resize(img_bgr, box, size=IMG_SIZE, pad_ratio=0.2):
    x, y, w, h = box
    pad = int(pad_ratio * max(w, h))
    x0 = max(x - pad, 0); y0 = max(y - pad, 0)
    x1 = min(x + w + pad, img_bgr.shape[1]); y1 = min(y + h + pad, img_bgr.shape[0])
    crop = img_bgr[y0:y1, x0:x1]
    if crop.size == 0:
        return None
    return cv2.resize(crop, size, interpolation=cv2.INTER_AREA)

def embed_facenet(model, face_bgr_160):
    rgb = cv2.cvtColor(face_bgr_160, cv2.COLOR_BGR2RGB)
    x = np.expand_dims(rgb, axis=0)  # (1,160,160,3)
    emb = model.embeddings(x)[0].astype(np.float32)
    return l2_normalize(emb)

def load_profiles(dataset_root: Path):
    """Return dict: {username: prototype_vector} for all users with profile.npz."""
    profiles = {}
    for user_dir in sorted(dataset_root.glob("*")):
        if not user_dir.is_dir():
            continue
        p = user_dir / "profile.npz"
        if p.exists():
            try:
                proto = np.load(str(p))["embedding_mean"].astype(np.float32)
                profiles[user_dir.name] = l2_normalize(proto)
            except Exception:
                pass
    return profiles

def capture_image_from_camera(camera_id=0):
    """Capture an image from camera. Returns BGR image or None."""
    print("[INFO] Opening camera... Press SPACE to capture, ESC to cancel.")
    cap = cv2.VideoCapture(camera_id)
    
    if not cap.isOpened():
        print("[ERR] Could not open camera.")
        return None
    
    # Set camera resolution (optional)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    captured_img = None
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERR] Failed to grab frame.")
            break
        
        # Display instructions on frame
        display = frame.copy()
        cv2.putText(display, "Press SPACE to capture, ESC to cancel", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow("Camera - Capture Image", display)
        
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            print("[INFO] Capture cancelled.")
            break
        elif key == 32:  # SPACE
            captured_img = frame.copy()
            print("[OK] Image captured!")
            break
    
    cap.release()
    cv2.destroyAllWindows()
    return captured_img

# --------------- Main -------------------
def main():
    print("=" * 50)
    print(" Multi-Factor Recognition System - Test")
    print("=" * 50)
    
    # Load model and profiles
    print("[INFO] Loading FaceNet model...")
    model = FaceNet()
    
    print("[INFO] Loading user profiles...")
    profiles = load_profiles(DATASET_ROOT)
    if not profiles:
        print(f"[ERR] No profiles found under {DATASET_ROOT}/<user>/profile.npz")
        return
    print(f"[OK] Loaded {len(profiles)} user profile(s): {', '.join(profiles.keys())}")

    # Capture image from camera
    img = capture_image_from_camera(CAMERA_ID)
    if img is None:
        print("[ERR] Failed to capture image from camera.")
        return

    # Detect face
    box = detect_largest_face_bgr(img)
    if box is None:
        print("[ERR] No face detected in the image.")
        return

    # Crop, resize, embed
    face_160 = crop_align_resize(img, box, size=IMG_SIZE, pad_ratio=0.2)
    if face_160 is None:
        print("[ERR] Empty crop after resizing.")
        return
    emb = embed_facenet(model, face_160)

    # Compare to all profiles
    best_user, best_sim = None, -1.0
    for username, proto in profiles.items():
        s = cosine_sim(emb, proto)
        if s > best_sim:
            best_sim = s
            best_user = username

    # Convert similarity to a simple percentage for display
    # Clamp to [0,1] then *100; this keeps it interpretable.
    match_pct = max(0.0, min(1.0, best_sim)) * 100.0

    # Decision
    decision = "Known" if best_sim >= TAU else "Unknown"

    # Report
    x, y, w, h = box
    print("\n" + "=" * 50)
    print(" Recognition Result")
    print("=" * 50)
    print(f"Detected face box: (x={x}, y={y}, w={w}, h={h})")
    print(f"Best match: {best_user}  |  Cosine similarity: {best_sim:.3f}  |  Match: {match_pct:.1f}%")
    print(f"Decision (@τ={TAU:.2f}): {decision}")
    print("=" * 50)

    # Optional: visualize and save a preview with label
    try:
        vis = img.copy()
        color = (0, 255, 0) if decision == "Known" else (0, 0, 255)
        cv2.rectangle(vis, (x, y), (x+w, y+h), color, 2)
        label = f"{best_user if decision=='Known' else 'Unknown'} ({match_pct:.0f}%)"
        cv2.putText(vis, label, (x, max(0, y-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        out_path = Path("recognition_preview.jpg")
        cv2.imwrite(str(out_path), vis)
        print(f"[OK] Saved preview -> {out_path}")
    except Exception:
        pass

if __name__ == "__main__":
    main()

