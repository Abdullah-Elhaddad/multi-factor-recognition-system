import os
import cv2
import numpy as np
from deepface import DeepFace

# CONFIGURATION 
DATA_DIR = os.path.join(os.path.dirname(__file__), "../data/faces")  # dataset path
DETECTOR_BACKEND = "mtcnn"  


# HELPERS 
def ensure_user_dir(username):
    """
    Create a folder for the user inside the dataset directory.
    """
    path = os.path.abspath(os.path.join(DATA_DIR, username))
    os.makedirs(path, exist_ok=True)
    return path


def save_face_image(face_img, out_dir, idx):
    """
    Save a face image (numpy array) safely after resizing and converting color.
    """
    if isinstance(face_img, np.ndarray):
        # Convert from float [0-1] to uint8 [0-255] if needed
        if face_img.max() <= 1.0:
            face_img = (face_img * 255).astype("uint8")

        # Ensure it’s 3-channel BGR for OpenCV
        if face_img.shape[-1] == 3:
            face_img = cv2.cvtColor(face_img, cv2.COLOR_RGB2BGR)

        face_img = cv2.resize(face_img, (160, 160))
        filename = os.path.join(out_dir, f"img_{idx:03d}.jpg")
        cv2.imwrite(filename, face_img)
        print(f"[INFO] Saved {filename}")
        return True
    return False


def capture(username, max_images=20):
    """
    Capture face images for a given user and store them in a dataset folder.
    """
    out_dir = ensure_user_dir(username)
    existing = len(os.listdir(out_dir))
    idx = existing
    print(f"[INFO] Starting capture for user '{username}'. Already have {existing} images.")
    print("Press SPACE to capture, or 'q' to quit.\n")

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  

    if not cap.isOpened():
        print("[ERROR] Cannot open camera.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to capture frame. Exiting...")
            break

        frame = cv2.flip(frame, 1)

        # Display current progress
        display = frame.copy()
        cv2.putText(display, f"User: {username}  Captured: {idx}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow("Face Capture (SPACE to capture)", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

        if key == 32:  # SPACEBAR pressed
            try:
                # DeepFace.extract_faces accepts a NumPy array directly
                faces = DeepFace.extract_faces(
                    img_path=frame,
                    detector_backend=DETECTOR_BACKEND,
                    enforce_detection=False
                )
            except Exception as e:
                print(f"[ERROR] Detector failed: {e}")
                faces = []

            if len(faces) > 0:
                face_img = faces[0]["face"]  # DeepFace already returns cropped region
                if save_face_image(face_img, out_dir, idx):
                    idx += 1
            else:
                print("[WARN] No face detected, try again.")

            if idx - existing >= max_images:
                print(f"[INFO] Reached maximum of {max_images} images.")
                break

    cap.release()
    cv2.destroyAllWindows()
    print(f"[INFO] Capture complete. Total images in '{out_dir}': {len(os.listdir(out_dir))}")


# MAIN ENTRY POINT 
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Capture faces using webcam and DeepFace detector.")
    parser.add_argument("username", help="Username for folder to save images")
    parser.add_argument("--max", type=int, default=20, help="Maximum number of images to capture")
    args = parser.parse_args()

    capture(args.username, max_images=args.max)
