import os
import io
import base64
import json
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from PIL import Image
import numpy as np
from deepface import DeepFace
from resemblyzer import VoiceEncoder, preprocess_wav
import librosa  # needed by resemblyzer internally

# CONFIG 
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FACES_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "data", "faces"))
DATA_VOICES_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "data", "voices"))
EMB_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "embeddings"))
PIN_FILE = os.path.abspath(os.path.join(BASE_DIR, "..", "data", "pins.json"))

MODEL_NAME = "Facenet"
DETECTOR_BACKEND = "opencv"
FACE_THRESHOLD = 0.80 
VOICE_THRESHOLD = 0.75 # voice verification is less effective, hence slightly lower threshold. 
MAX_ATTEMPTS = 3

os.makedirs(DATA_FACES_DIR, exist_ok=True)
os.makedirs(DATA_VOICES_DIR, exist_ok=True)
os.makedirs(EMB_DIR, exist_ok=True)
os.makedirs(os.path.dirname(PIN_FILE), exist_ok=True)

# initialize PIN file if not exists
if not os.path.exists(PIN_FILE):
    with open(PIN_FILE, "w") as f:
        json.dump({}, f)

LOGIN_ATTEMPTS = {}

# single global encoder
voice_encoder = VoiceEncoder("cpu")

app = Flask(__name__, static_folder="../frontend", static_url_path="/")
CORS(app)


# UTILS 
def save_base64_image(b64_str, out_path):
    """Save a base64-encoded image to disk as JPG/PNG."""
    header_removed = b64_str.split(",")[-1]
    img_bytes = base64.b64decode(header_removed)
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    img.save(out_path)


def save_base64_audio(b64_str, out_path):
    """Save WAV audio sent from browser."""
    header_removed = b64_str.split(",")[-1]
    audio_bytes = base64.b64decode(header_removed)
    with open(out_path, "wb") as f:
        f.write(audio_bytes)


def image_to_embedding(image_path):
    """Compute DeepFace embedding for an image."""
    rep = DeepFace.represent(
        img_path=image_path,
        model_name=MODEL_NAME,
        detector_backend=DETECTOR_BACKEND,
        enforce_detection=False
    )

    if rep is None:
        return None

    # DeepFace return formats
    if isinstance(rep, list):
        if isinstance(rep[0], dict) and "embedding" in rep[0]:
            return np.array(rep[0]["embedding"])
        return np.array(rep[0])
    if isinstance(rep, dict) and "embedding" in rep:
        return np.array(rep["embedding"])

    return np.array(rep)


def cosine_similarity(a, b):
    return float(np.dot(a, b) /
                 (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))


def load_pins():
    with open(PIN_FILE, "r") as f:
        return json.load(f)


def save_pins(pins):
    with open(PIN_FILE, "w") as f:
        json.dump(pins, f, indent=2)


def inc_attempts(pin):
    LOGIN_ATTEMPTS[pin] = LOGIN_ATTEMPTS.get(pin, 0) + 1
    remaining = max(0, MAX_ATTEMPTS - LOGIN_ATTEMPTS[pin])
    locked = LOGIN_ATTEMPTS[pin] >= MAX_ATTEMPTS
    return LOGIN_ATTEMPTS[pin], remaining, locked


def reset_attempts(pin):
    LOGIN_ATTEMPTS.pop(pin, None)


#  VOICE EMBEDDING 
def build_voice_embedding(username):
    """
    Builds and saves a WAV-based voice embedding.
    Expects: DATA_VOICES_DIR/<username>.wav
    """
    wav_path = os.path.join(DATA_VOICES_DIR, f"{username}.wav")
    if not os.path.exists(wav_path):
        print("No WAV file for:", username)
        return None

    try:
        wav = preprocess_wav(wav_path)
        emb = voice_encoder.embed_utterance(wav)
    except Exception as e:
        print("Error building voice embedding:", e)
        return None

    out_path = os.path.join(EMB_DIR, f"{username}_voice.npy")
    np.save(out_path, emb)
    print("Saved voice embedding for", username)
    return emb


# ROUTES 
@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.route("/congrats.html")
def congrats_page():
    return send_from_directory(app.static_folder, "congrats.html")


@app.route("/users", methods=["GET"])
def list_users():
    users = [d for d in os.listdir(DATA_FACES_DIR)
             if os.path.isdir(os.path.join(DATA_FACES_DIR, d))]
    return jsonify(users)


#  REGISTER FACE + PIN 
@app.route("/register", methods=["POST"])
def register():
    data = request.get_json()
    username = data.get("username")
    pin = data.get("pin")
    b64 = data.get("image")

    if not username or not pin or not b64:
        return jsonify({"error": "username, pin, and image required"}), 400

    user_dir = os.path.join(DATA_FACES_DIR, username)
    os.makedirs(user_dir, exist_ok=True)

    next_idx = len([x for x in os.listdir(user_dir)
                    if x.lower().endswith(".jpg")]) + 1

    filename = os.path.join(user_dir, f"img_{next_idx:03d}.jpg")
    save_base64_image(b64, filename)

    pins = load_pins()
    pins[username] = pin
    save_pins(pins)

    return jsonify({"saved": filename, "pin_stored": True})


# REGISTER VOICE 
@app.route("/register_voice", methods=["POST"])
def register_voice():
    """
    Body: {"username": "haddad", "pin": "1234", "audio": "data:audio/wav;base64,..."}
    Saves: DATA_VOICES_DIR/<username>.wav
    """
    data = request.get_json()
    username = data.get("username")
    pin = data.get("pin")
    audio_b64 = data.get("audio")

    if not username or not pin or not audio_b64:
        return jsonify({"error": "username, pin, and audio required"}), 400

    pins = load_pins()
    if pins.get(username) != pin:
        return jsonify({"error": "PIN does not match this username"}), 403

    wav_path = os.path.join(DATA_VOICES_DIR, f"{username}.wav")
    save_base64_audio(audio_b64, wav_path)

    emb = build_voice_embedding(username)
    if emb is None:
        return jsonify({"error": "Could not build voice embedding"}), 500

    return jsonify({"voice_saved": True})


# BUILD FACE EMBEDDINGS 
@app.route("/build_embeddings", methods=["POST"])
def build_embeddings():
    """
    Optional JSON body: {"username": "haddad"}
    - If username provided -> build embeddings only for that user
    - If not provided -> build for all users
    """
    payload = request.get_json(silent=True) or {}
    target_username = payload.get("username")

    results = {}

    for username in os.listdir(DATA_FACES_DIR):
        user_dir = os.path.join(DATA_FACES_DIR, username)
        if not os.path.isdir(user_dir):
            continue

        if target_username and username != target_username:
            continue

        files = [os.path.join(user_dir, f)
                 for f in os.listdir(user_dir)
                 if f.lower().endswith(".jpg")]

        if not files:
            results[username] = "no_images"
            continue

        embs = []
        for f in files:
            try:
                emb = image_to_embedding(f)
                if emb is not None:
                    embs.append(emb)
            except Exception as e:
                print("Embedding error:", f, e)

        if not embs:
            results[username] = "failed"
            continue

        embs = np.vstack(embs)
        mean_emb = embs.mean(axis=0)

        np.save(os.path.join(EMB_DIR, f"{username}_mean.npy"), mean_emb)
        np.save(os.path.join(EMB_DIR, f"{username}_all.npy"), embs)

        results[username] = {"n_images": len(embs)}

    if target_username and target_username not in results:
        results[target_username] = "user_not_found_or_no_images"

    return jsonify(results)


# VERIFY FACE (PIN + FACE) 
@app.route("/verify", methods=["POST"])
def verify_face():
    data = request.get_json()
    pin = data.get("pin")
    b64 = data.get("image")

    if not pin or not b64:
        return jsonify({"error": "pin and image required"}), 400

    pins = load_pins()

    username = None
    for user, stored_pin in pins.items():
        if stored_pin == pin:
            username = user
            break

    if not username:
        attempts, remaining, locked = inc_attempts(pin)
        return jsonify({
            "error": "Invalid PIN",
            "attempts": attempts,
            "remaining": remaining,
            "locked": locked
        }), 403

    tmp_path = os.path.join(EMB_DIR, "__probe_face.jpg")
    save_base64_image(b64, tmp_path)

    probe_emb = image_to_embedding(tmp_path)
    if probe_emb is None:
        attempts, remaining, locked = inc_attempts(pin)
        return jsonify({
            "error": "Could not compute face embedding",
            "attempts": attempts,
            "remaining": remaining,
            "locked": locked
        }), 500

    face_emb_path = os.path.join(EMB_DIR, f"{username}_mean.npy")
    if not os.path.exists(face_emb_path):
        attempts, remaining, locked = inc_attempts(pin)
        return jsonify({
            "error": "Face data missing. Build embeddings.",
            "attempts": attempts,
            "remaining": remaining,
            "locked": locked
        }), 400

    user_emb = np.load(face_emb_path)
    score = cosine_similarity(probe_emb, user_emb)
    face_ok = score >= FACE_THRESHOLD

    if not face_ok:
        attempts, remaining, locked = inc_attempts(pin)
        return jsonify({
            "username": username,
            "face_verified": False,
            "score": score,
            "threshold": FACE_THRESHOLD,
            "attempts": attempts,
            "remaining": remaining,
            "locked": locked
        })

    # success
    return jsonify({
        "username": username,
        "face_verified": True,
        "score": score,
        "threshold": FACE_THRESHOLD,
        "attempts": LOGIN_ATTEMPTS.get(pin, 0),
        "remaining": MAX_ATTEMPTS - LOGIN_ATTEMPTS.get(pin, 0),
        "locked": False
    })


# VERIFY VOICE 
@app.route("/verify_voice", methods=["POST"])
def verify_voice():
    """
    Body: {"pin":"1234", "audio":"data:audio/wav;base64,..."}
    """
    data = request.get_json()
    pin = data.get("pin")
    audio_b64 = data.get("audio")

    if not pin or not audio_b64:
        return jsonify({"error": "pin and audio required"}), 400

    pins = load_pins()

    username = None
    for user, stored_pin in pins.items():
        if stored_pin == pin:
            username = user
            break

    if not username:
        attempts, remaining, locked = inc_attempts(pin)
        return jsonify({
            "error": "Invalid PIN",
            "attempts": attempts,
            "remaining": remaining,
            "locked": locked
        }), 403

    probe_path = os.path.join(DATA_VOICES_DIR, "__probe.wav")
    save_base64_audio(audio_b64, probe_path)

    stored_emb_path = os.path.join(EMB_DIR, f"{username}_voice.npy")
    if not os.path.exists(stored_emb_path):
        attempts, remaining, locked = inc_attempts(pin)
        return jsonify({
            "error": "No stored voice for this user",
            "attempts": attempts,
            "remaining": remaining,
            "locked": locked
        }), 400

    stored_emb = np.load(stored_emb_path)

    try:
        wav = preprocess_wav(probe_path)
        probe_emb = voice_encoder.embed_utterance(wav)
    except Exception as e:
        attempts, remaining, locked = inc_attempts(pin)
        return jsonify({
            "error": f"Voice embedding failed: {e}",
            "attempts": attempts,
            "remaining": remaining,
            "locked": locked
        }), 500

    score = cosine_similarity(probe_emb, stored_emb)
    voice_ok = score >= VOICE_THRESHOLD

    if not voice_ok:
        attempts, remaining, locked = inc_attempts(pin)
        return jsonify({
            "username": username,
            "voice_verified": False,
            "score": score,
            "threshold": VOICE_THRESHOLD,
            "attempts": attempts,
            "remaining": remaining,
            "locked": locked
        })

    # MFA success
    reset_attempts(pin)
    return jsonify({
        "username": username,
        "voice_verified": True,
        "score": score,
        "threshold": VOICE_THRESHOLD,
        "attempts": 0,
        "remaining": MAX_ATTEMPTS,
        "locked": False,
        "mfa_success": True
    })


# ----------------- MAIN -----------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
