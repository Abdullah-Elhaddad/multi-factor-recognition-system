"""
collect_images.py
- Input: username
- Opens webcam, detects largest face, press ENTER to capture 12 images
- Saves resized crops to: dataset/<username>/images/image.<n>.jpg (160x160)
"""

import cv2
from pathlib import Path

IMG_SIZE = (160, 160)
NUM_IMAGES = 12
DATASET_ROOT = Path("dataset")

def detect_largest_face(gray, face_cascade):
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    if len(faces) == 0:
        return None
    return tuple(max(faces, key=lambda r: r[2]*r[3]))  # <-- convert to tuple

def main():
    username = input("Username to collect images for: ").strip()
    if not username:
        print("[ERR] Username is required."); return

    user_img_dir = DATASET_ROOT / username / "images"
    user_img_dir.mkdir(parents=True, exist_ok=True)

    cascade = cv2.CascadeClassifier(cv2.data.haarcascades +
                                    "haarcascade_frontalface_default.xml")
    if cascade.empty():
        print("[ERR] Could not load Haar cascade."); return

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERR] Cannot open webcam."); return

    print("[INFO] Press ENTER to capture each image (need 12). Press 'q' to quit.\n")
    captured = 0
    try:
        while captured < NUM_IMAGES:
            ok, frame = cap.read()
            if not ok: print("[ERR] Camera read failed."); break

            preview = frame.copy()
            gray = cv2.cvtColor(preview, cv2.COLOR_BGR2GRAY)
            box = detect_largest_face(gray, cascade)

            msg = f"Face: {'FOUND' if box else 'NOT FOUND'} | ENTER to capture ({captured+1}/{NUM_IMAGES})"
            cv2.putText(preview, msg, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

            if box:
                x,y,w,h = box
                cv2.rectangle(preview, (x,y), (x+w,y+h), (0,255,0), 2)

            cv2.imshow("Collect Images", preview)
            key = cv2.waitKey(1) & 0xFF

            if key in (ord('q'), ord('Q')):
                print("[INFO] Quit requested."); break
            if key in (13, 10):  # ENTER
                if not box:
                    print("[WARN] No face detected. Try again."); continue

                x,y,w,h = box
                pad = int(0.2 * max(w,h))
                x0 = max(x - pad, 0); y0 = max(y - pad, 0)
                x1 = min(x + w + pad, frame.shape[1]); y1 = min(y + h + pad, frame.shape[0])
                crop = frame[y0:y1, x0:x1]
                if crop.size == 0:
                    print("[WARN] Empty crop. Try again."); continue

                crop = cv2.resize(crop, IMG_SIZE, interpolation=cv2.INTER_AREA)
                out_path = user_img_dir / f"image.{captured+1}.jpg"
                cv2.imwrite(str(out_path), crop)
                print(f"[OK] Saved -> {out_path}")
                captured += 1

        if captured == NUM_IMAGES:
            print("[DONE] Collected all images.")
        else:
            print(f"[PARTIAL] Collected {captured}/{NUM_IMAGES}.")
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
