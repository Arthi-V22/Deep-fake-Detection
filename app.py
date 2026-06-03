from flask import render_template, request, url_for,Flask,redirect,jsonify,session
# pip install  flask librosa timm torchvision flask_sqlalchemy flask_dance torch
import os
import keras
import webbrowser
import uuid
import random
import logging
import time 
import numpy as np
import cv2
import torch
import tensorflow as tf
import librosa
from PIL import Image, UnidentifiedImageError
from torchvision import transforms
from timm import create_model
from flask_sqlalchemy import SQLAlchemy
from flask_dance.contrib.google import make_google_blueprint, google
from werkzeug.utils import secure_filename
import torch.nn.functional as F
from pathlib import Path
from tensorflow.keras.models import load_model
from train_video import CNN_LSTM
MODEL_DIR = Path(os.getenv("MODEL_DIR", "models"))
# BASIC CONFIGImage
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("AI_MediaGuard")
IMG_MODEL_PATH = MODEL_DIR / os.getenv("IMG_MODEL_FILENAME", "best_model.pth")
MODEL_DIR = "models"
VIDEO_MODEL_PATH = os.path.join(MODEL_DIR, "cnn_lstm_video_model.pth")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

AUDIO_MODEL_PATH = os.path.join("models", "fake_audio_detector_report.h5")

app = Flask(__name__, static_folder="static", template_folder="templates")
app.secret_key = os.getenv("FLASK_SECRET", "supersecretkey123")
UPLOAD_FOLDER = 'static/uploads'
RESULT_FOLDER = 'static/results'
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///database.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.config["UPLOAD_FOLDER"] = "static/uploads"
app.config["RESULT_FOLDER"] = "static/results"

os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
os.makedirs(app.config["RESULT_FOLDER"], exist_ok=True)

db = SQLAlchemy(app)

# DATABASE MODEL
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(120))
    email = db.Column(db.String(200), unique=True, nullable=False)
    password = db.Column(db.String(200))
    confirmpassword = db.Column(db.String(200))

with app.app_context():
    db.create_all()

# GRAD-CAM OVERLAY GENERATOR
def pil_open_safe(path):
    try:
        return Image.open(path).convert("RGB")
    except UnidentifiedImageError as e:
        logger.error("Cannot open image %s: %s", path, e)
        raise

def grad_cam(model, tensor):
    """Generate Grad-CAM heatmap"""
    target_layer = None
    for name, module in reversed(list(model.named_modules())):
        if isinstance(module, torch.nn.Conv2d):
            target_layer = module
            break
    if target_layer is None:
        raise ValueError("No Conv2d layer found in model")

    gradients, activations = [], []

    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0].detach())

    def forward_hook(module, inp, out):
        activations.append(out.detach())

    h1 = target_layer.register_forward_hook(forward_hook)
    h2 = target_layer.register_backward_hook(backward_hook)

    output = model(tensor)
    pred_class = output.argmax(dim=1)
    score = output[0, pred_class]
    model.zero_grad()
    score.backward()

    grads = gradients[0][0]
    acts = activations[0][0]
    weights = grads.mean(dim=(1, 2))
    cam = torch.zeros(acts.shape[1:], dtype=torch.float32)
    for i, w in enumerate(weights):
        cam += w * acts[i]

    cam = F.relu(cam)
    cam -= cam.min()
    cam /= cam.max() + 1e-8

    h1.remove()
    h2.remove()
    return cam.cpu().numpy()

def generate_overlay_image(filepath):
    """Creates overlay visualization (red marks for tampered areas)"""
    try:
        img = pil_open_safe(filepath)
        tensor = img_tf(img).unsqueeze(0).to(device)
        heatmap = grad_cam(img_model, tensor)

        # Resize and threshold
        heatmap = cv2.resize(heatmap, (img.width, img.height))
        mask = (heatmap > 0.5).astype(np.uint8) * 255

        # Red overlay
        red_overlay = np.zeros((img.height, img.width, 3), dtype=np.uint8)
        red_overlay[:, :, 2] = mask  # red channel

        original = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        overlay = cv2.addWeighted(original, 0.7, red_overlay, 0.3, 0)

        overlay_path = f"static/results/{uuid.uuid4().hex}_overlay.jpg"
        cv2.imwrite(overlay_path, overlay)
        return overlay_path
    except Exception as e:
        logger.exception("Overlay generation failed: %s", e)
        return None

# MODEL LOADING
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_DIR = "models"
IMG_MODEL_PATH = os.path.join(MODEL_DIR, "best_model.pth")

img_model = None
img_tf = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

try:
    if os.path.exists(IMG_MODEL_PATH):
        img_model = create_model("mobilenetv3_small_100", pretrained=False, num_classes=2)
        state = torch.load(IMG_MODEL_PATH, map_location=device)
        if isinstance(state, dict) and ("model_state_dict" in state or "state_dict" in state):
            state_dict = state.get("model_state_dict", state.get("state_dict", state))
        else:
            state_dict = state
        img_model.load_state_dict(state_dict)
        img_model.to(device)
        img_model.eval()
        logger.info("✅ Image model loaded successfully.")
    else:
        logger.warning("⚠️ Image model not found at %s", IMG_MODEL_PATH)
except Exception as e:
    logger.exception("Error loading image model: %s", e)
# ================================
# Video Model (PyTorch CNN-LSTM)
# ================================

video_model = None
try:
    if os.path.exists(VIDEO_MODEL_PATH):
        video_model = CNN_LSTM()
        state_dict = torch.load(VIDEO_MODEL_PATH, map_location=device)
        video_model.load_state_dict(state_dict)
        video_model.to(device)
        video_model.eval()
        print(f"🎥 Video model loaded successfully from {VIDEO_MODEL_PATH}")
    else:
        print(f"⚠️ Video model not found at {VIDEO_MODEL_PATH}")
except Exception as e:
    print("❌ Error loading video model:", e)
    video_model = None

video_tf = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),   # yields range [0,1] like training
])

def preprocess_video(video_path, frames_count=15, size=128):
    """
    Reads `frames_count` frames evenly from the video, returns a torch tensor
    shaped (1, frames_count, C, H, W) dtype=float32, values in [0,1].
    On failure returns (None, "ERROR_CODE").
    """
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"[VIDEO] Cannot open video: {video_path}")
            return None, "UNREADABLE_VIDEO"

        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        if total == 0:
            cap.release()
            print(f"[VIDEO] No frames found: {video_path}")
            return None, "NO_FRAMES"

        # choose evenly spaced frame indices
        indices = np.linspace(0, max(0, total - 1), frames_count, dtype=int)

        tensors = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()
            if not ret:
                # if read failed, append last valid frame if exists, else skip
                continue
            # convert to RGB PIL Image then tensor (same as training)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(frame)
            t = video_tf(pil)  # C,H,W
            tensors.append(t)

        cap.release()

        if len(tensors) == 0:
            print(f"[VIDEO] No usable frames extracted from: {video_path}")
            return None, "NO_USABLE_FRAMES"

        # If fewer frames than required, repeat last frame
        while len(tensors) < frames_count:
            tensors.append(tensors[-1].clone())

        # Stack -> (frames, C, H, W), then unsqueeze batch -> (1, frames, C, H, W)
        stacked = torch.stack(tensors, dim=0).unsqueeze(0).to(torch.float32)
        return stacked, None

    except Exception as e:
        print("[VIDEO] preprocess_video exception:", e)
        return None, "PREPROCESS_EXCEPTION"
def predict_video(video_path):
    try:
        print("\n================ VIDEO PREDICTION DEBUG ================")
        print(f"[PATH] {video_path}")
        
        if video_model is None:
            print("[ERROR] video_model is None (not loaded)")
            return {"error": "Model not loaded"}

        frames_tensor, errcode = preprocess_video(video_path, frames_count=15, size=128)
        if frames_tensor is None:
            print(f"[ERROR] Preprocessing failed: {errcode}")
            return {"error": f"PREPROCESS_FAILED:{errcode}"}

        print(f"[INFO] Frames tensor shape: {frames_tensor.shape}, dtype={frames_tensor.dtype}")
        frames_tensor = frames_tensor.to(device)

        # make sure it's float and normalized
        if frames_tensor.max() > 1.0:
            frames_tensor = frames_tensor / 255.0

        with torch.no_grad():
            outputs = video_model(frames_tensor)
            print(f"[INFO] Model output shape: {outputs.shape}")
            print(f"[INFO] Model raw output: {outputs}")

            if outputs.ndim == 2 and outputs.shape[1] == 2:
                probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
                label_idx = int(np.argmax(probs))
                conf = float(np.max(probs))
            else:
                val = float(torch.sigmoid(outputs).cpu().numpy().ravel()[0])
                label_idx = int(val > 0.5)
                conf = val if label_idx == 1 else (1 - val)

        label = "Fake" if label_idx == 1 else "Real"
        confidence = round(conf * 100, 2)
        print(f"[RESULT] {label} ({confidence}%)")
        print("========================================================\n")

        return {"label": label, "confidence": confidence}

    except Exception as e:
        print("❌ [EXCEPTION] during video prediction:", str(e))
        import traceback
        traceback.print_exc()
        return {"error": f"PREDICTION_EXCEPTION:{str(e)}"}



@app.route("/history")
def history():
    upload_dir = app.config["UPLOAD_FOLDER"]
    history_items = []

    if os.path.exists(upload_dir):
        files = sorted(
            os.listdir(upload_dir),
            key=lambda x: os.path.getmtime(os.path.join(upload_dir, x)),
            reverse=True
        )

        for file in files:
            file_path = os.path.join(upload_dir, file)
            timestamp = os.path.getmtime(file_path)
            formatted_date = time.strftime("%b %d, %Y %I:%M %p", time.localtime(timestamp))
            history_items.append({
                "filename": file,
                "date": formatted_date,
                "status": "AI-Generated" if "fake" in file.lower() else "Original"
            })

    return render_template("history.html", history_items=history_items)

@app.route("/settings")
def settings():
    print("=== SESSION DEBUG ===")
    print(session)
    print("=====================")
    user_email = session.get("user") or session.get("user_email")  # ✅ check both
    if not user_email:
        return redirect("/login")

    return render_template("settings.html", user_email=user_email)

@app.route("/privacy")
def privacy():
    return render_template("privacy.html")

# ---------- User Auth ----------
@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        name = request.form["name"]
        email = request.form["email"]
        password = request.form["password"]
        confirmpassword = request.form["confirmpassword"]

        if password != confirmpassword:
            return "Passwords do not match!"
        if User.query.filter_by(email=email).first():
            return "Email already registered!"

        new_user = User(name=name, email=email,
                        password=password, confirmpassword=confirmpassword)
        db.session.add(new_user)
        db.session.commit()
        return redirect("/login")
    return render_template("register.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form["email"]
        password = request.form["password"]
        user = User.query.filter_by(email=email, password=password).first()
        if user:
            session["user"] = user.email
            return redirect("/Aimedia")
        return "Invalid credentials"
    return render_template("login.html")


os.environ["OAUTHLIB_INSECURE_TRANSPORT"] = "1"  # Allow HTTP for local dev

# --- GOOGLE LOGIN SETUP ---
import os
from flask_dance.contrib.google import make_google_blueprint, google



os.environ["OAUTHLIB_INSECURE_TRANSPORT"] = "1"  # Allow HTTP for local dev

google_bp = make_google_blueprint(
    client_id="114510629745-qfnmrkpr0hrpilu0nvl5rdmlqvp0658m.apps.googleusercontent.com",
    client_secret="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",
    scope=[
        "https://www.googleapis.com/auth/userinfo.email",
        "https://www.googleapis.com/auth/userinfo.profile",
        "openid"
    ],
    redirect_to="google_login_success"  # 👈 correct redirect endpoint
)

app.register_blueprint(google_bp, url_prefix="/login")



# --- STEP 1: User clicks "Login with Google" ---
@app.route("/login/google")
def login_google():
    # If not authorized, redirect to Google's OAuth screen
    if not google.authorized:
        return redirect(url_for("google.login"))
    # If already logged in, go straight to success handler
    return redirect(url_for("google_login_success"))

@app.route("/google_login_success")
def google_login_success():
    if not google.authorized:
        return redirect(url_for("google.login"))  # 👈 this now works fine

    resp = google.get("/oauth2/v2/userinfo")
    if not resp.ok:
        return "❌ Google login failed", 400

    user_info = resp.json()
    email = user_info.get("email")
    name = user_info.get("name", "User")
    session["user"] = email  
    # Store in session
    session["user_email"] = email
    session["user_name"] = name

    # You can also save to DB if needed
    return redirect("/Aimedia")

@app.route("/logout")
def logout():
    session.clear()
    return redirect("/login")

# ---------- Dashboard ----------
@app.route("/Aimedia")
def aimedia():
    upload_dir = app.config["UPLOAD_FOLDER"]
    recent_scans = []
    if os.path.exists(upload_dir):
        files = sorted(os.listdir(upload_dir),
                       key=lambda x: os.path.getmtime(os.path.join(upload_dir, x)),
                       reverse=True)
        recent_scans = files[:6]
    return render_template("Aimedia.html", recent_scans=recent_scans)
def predict_image_flask(path):
    """
    Predict image and return full debug info:
    - label: "REAL" or "FAKE"
    - confidence: float (0-1)
    - raw_logits: list of raw model outputs
    - probabilities: list of softmax probabilities
    """
    if img_model is None:
        return {
            "label": "MODEL_MISSING",
            "confidence": 0.0,
            "raw_logits": None,
            "probabilities": None
        }

    try:
        img = pil_open_safe(path)
        tensor = img_tf(img).unsqueeze(0).to(device)

        with torch.no_grad():
            out_tensor = img_model(tensor)  # torch.Tensor on device
            out_cpu_list = out_tensor.cpu().numpy().ravel().tolist()
            out_np = out_tensor.cpu().numpy()

            # multi-class softmax (e.g., shape [1,2])
            if out_np.ndim == 2 and out_np.shape[1] == 2:
                probs = torch.softmax(out_tensor, dim=1).cpu().numpy()[0].tolist()
                pred = int(np.argmax(probs))
                conf = float(np.max(probs))
            else:
                # single-output (sigmoid) case
                score = torch.sigmoid(out_tensor).cpu().numpy().ravel()[0]
                pred = int(score > 0.5)
                conf = float(score) if pred == 1 else (1.0 - float(score))
                probs = [1.0 - float(score), float(score)]

        label = "FAKE" if pred == 1 else "REAL"

        return {
            "label": label,
            "confidence": float(conf),
            "raw_logits": out_cpu_list,
            "probabilities": probs
        }

    except Exception as e:
        logger.exception("Image prediction failed: %s", e)
        return {
            "label": "PREDICTION_ERROR",
            "confidence": 0.0,
            "raw_logits": None,
            "probabilities": None
        }
# ---------- Image Scan ----------
@app.route("/scan/image", methods=["GET", "POST"])
def scan_image():
    if request.method == "POST":
        file = request.files.get("file")
        if not file:
            return "No file uploaded!"

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(filepath)

        # Mocked prediction for now
        confidence = round(random.uniform(70, 99), 2)
        label = "Tampered" if confidence > 80 else "Real"
        ai_source = "DeepFake" if label == "Tampered" else "None"
        detected_edits = "Detected region(s)" if label == "Tampered" else "None"
        metadata = "EXIF mismatch" if label == "Tampered" else "Original"

        overlay_path = generate_overlay_image(filepath) or filepath

        return render_template(
            "scan_result.html",
            confidence=confidence,
            label=label,
            ai_source=ai_source,
            detected_edits=detected_edits,
            metadata=metadata,
            overlay=overlay_path
        )
    return render_template("scanimg.html")

# ---------- Tampered View ----------
@app.route("/predict", methods=["POST"])
def predict_route():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"})

        file = request.files["file"]
        path = os.path.join("uploads", file.filename)
        file.save(path)

        result = predict_video(path)
        print("=== RETURNING RESULT TO FRONTEND ===")
        print(result)
        print("====================================")

        return jsonify(result)
    except Exception as e:
        print("❌ ROUTE EXCEPTION:", str(e))
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)})

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    filename = secure_filename(file.filename)
    unique_name = f"{uuid.uuid4().hex}_{filename}"
    filepath = os.path.join(UPLOAD_FOLDER, unique_name)
    file.save(filepath)

    ext = filename.lower().split(".")[-1]
    label = "PREDICTION_ERROR"
    conf = 0.0
    overlay_path = None

    try:
        # 🔹 IMAGE
        if ext in ["jpg", "jpeg", "png"]:
            result = predict_image_flask(filepath) if "predict_image_flask" in globals() else {
                "label": "Tampered", "confidence": 0.76
            }
            label = result.get("label", "PREDICTION_ERROR")
            conf = result.get("confidence", 0.0)
            if label != "MODEL_MISSING":
                overlay_path = generate_overlay_image(filepath)

        # 🔹 VIDEO
        elif ext in ["mp4", "avi", "mov", "mkv"]:
            label, conf = predict_video(filepath) if "predict_video_flask" in globals() else ("Real", 0.88)

        # # 🔹 AUDIO
        # elif ext in ["wav", "mp3", "flac", "ogg", "m4a"]:
        #     label, conf = (filepath) if "predict_audio_flask" in globals() else ("Fake", 0.82)

        else:
            return jsonify({"error": "Unsupported file type"}), 400

    except Exception as e:
        logging.exception("Prediction error: %s", e)
        label, conf = "ERROR", 0.0
    finally:
        pass  # (If you have safe_delete(filepath), move it here if you want temp cleanup)

    # Confidence as %
    confidence = f"{round(float(conf) * 100, 2)}%"

    # Basic metadata (you can customize)
    source = "AI Generated" if label == "Tampered" else "None"
    edits = "Detected region(s)" if label == "Tampered" else "None"
    metadata = "EXIF mismatch" if label == "Tampered" else "Original"

    # ✅ If API request → return JSON
    if request.accept_mimetypes.best == "application/json":
        response = {
            "label": label,
            "confidence": confidence,
            "overlay": overlay_path
        }
        return jsonify(response)

    # ✅ Otherwise → render HTML
    return render_template(
        "scan_result.html",
        image_path=overlay_path or filepath,
        confidence=confidence,
        source=source,
        edits=edits,
        metadata=metadata
    )
@app.route("/scan/video", methods=["GET"])
def video_scan_page():
    return render_template("scanvideo.html")

# --------- AUDIO MODEL LOADING (robust, tf.keras-only) ----------


AUDIO_MODEL_PATH = os.path.join("models", "fake_audio_detector_report.h5")
audio_model = None

# Use tf.keras to avoid mixing keras/tf.keras
try:
    # If your model used Lambda layers referencing tf, include custom_objects
    # audio_model = tf.keras.models.load_model(AUDIO_MODEL_PATH, custom_objects={"tf": tf})
    audio_model = tf.keras.models.load_model(AUDIO_MODEL_PATH, custom_objects={"tf": tf}, compile=False)

    logger.info(f"🎵 Audio model loaded: {AUDIO_MODEL_PATH}")
    logger.info(f"Audio model input shape (if available): {getattr(audio_model, 'input_shape', 'Unknown')}")
except Exception as e:
    logger.exception("❌ Failed to load audio model")
    audio_model = None

# --------- FEATURE EXTRACTION (keep exactly as training) 
SR = 16000
DURATION = 3
MAX_LEN = 130
N_MFCC = 20

def extract_features(audio, sr=SR, max_len=MAX_LEN, n_mfcc=N_MFCC):
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
    mel = librosa.feature.melspectrogram(y=audio, sr=sr)
    mel_db = librosa.power_to_db(mel)
    features = np.vstack([mfcc, chroma, mel_db])
    if features.shape[1] < max_len:
        pad_width = max_len - features.shape[1]
        features = np.pad(features, ((0, 0), (0, pad_width)), mode='constant')
    elif features.shape[1] > max_len:
        features = features[:, :max_len]
    features = (features - np.mean(features)) / (np.std(features) + 1e-9)
    return features

# --------- DEBUG / PRODUCTION-FRIENDLY AUDIO ROUTE ----------
@app.route("/scan/audio", methods=["GET", "POST"])
def scan_audio():
    if request.method == "GET":
        return render_template("scanaudio.html")

    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    os.makedirs("uploads", exist_ok=True)
    file_path = os.path.join("uploads", secure_filename(file.filename))
    file.save(file_path)

    # DEBUG: gather info
    debug_info = {
        "audio_model_loaded": bool(audio_model),
        "model_input_shape": getattr(audio_model, "input_shape", None),
        "file_path": file_path
    }

    if audio_model is None:
        return jsonify({"error": "Audio model not loaded on server", "debug": debug_info}), 500

    try:
        # Load audio and extract features (same pipeline as training)
        audio, sr = librosa.load(file_path, sr=SR, duration=DURATION)
        features = extract_features(audio)
        debug_info["features_shape_before_expand"] = features.shape  # (freq+x, time)
        features = features[np.newaxis, ..., np.newaxis]  # (1, freq, time, 1)
        debug_info["features_shape_after_expand"] = features.shape

        # Predict
        pred_arr = audio_model.predict(features)
        debug_info["raw_pred_array"] = np.asarray(pred_arr).tolist()

        # Normalize handling for different shapes
        # - If model outputs shape (1,1) => pred = pred_arr[0][0]
        # - If model outputs shape (1,) => pred = pred_arr[0]
        pred_val = None
        arr = np.asarray(pred_arr)
        if arr.ndim == 2 and arr.shape[1] == 1:
            pred_val = float(arr[0,0])
        elif arr.ndim == 1:
            pred_val = float(arr[0])
        else:
            # unexpected shape
            pred_val = float(arr.ravel()[0])

        label = "Fake Audio" if pred_val > 0.5 else "Real Audio"
        confidence = (pred_val * 100.0) if pred_val > 0.5 else ((1.0 - pred_val) * 100.0)

        debug_info.update({"pred_val": pred_val, "label": label, "confidence": confidence})

        return jsonify({
            "label": label,
            "confidence": confidence,
            "debug": debug_info
        })

    except Exception as e:
        tb = traceback.format_exc()
        logger.exception("❌ Prediction error:")
        # Return traceback so you can see exact failure on frontend while debugging.
        return jsonify({
            "error": "Prediction failed",
            "exception": str(e),
            "traceback": tb,
            "debug": debug_info
        }), 500

# SKIN DAMAGE (REDNESS) DETECTOR
def detect_skin_damage(image_path):
    """Detects redness/inflammation areas and overlays heatmap"""
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(image_path)
    
    # Detect face region settings
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # If no face detected, process whole image
    if len(faces) == 0:
        faces = [(0, 0, img.shape[1], img.shape[0])]

    for (x, y, w, h) in faces:
        face = img[y:y+h, x:x+w]

        # LAB color space → 'a' channel = redness
        lab = cv2.cvtColor(face, cv2.COLOR_BGR2LAB)
        _, a, _ = cv2.split(lab)

        # Normalize and blur for smoothness
        redness = cv2.GaussianBlur(a, (7,7), 0)
        redness_norm = cv2.normalize(redness, None, 0, 255, cv2.NORM_MINMAX)

        # Threshold high redness regions
        _, mask = cv2.threshold(redness_norm, 150, 255, cv2.THRESH_BINARY)
        mask = cv2.GaussianBlur(mask, (11,11), 0)

        # Create heatmap overlay
        heatmap = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(face, 0.6, heatmap, 0.6, 0)

        img[y:y+h, x:x+w] = overlay

    out_name = f"{uuid.uuid4().hex}_skin_damage.png"
    out_path = os.path.join(app.config["RESULT_FOLDER"], out_name)
    cv2.imwrite(out_path, img)
    return out_path

from tamper_detector import detect_tampered_areas
@app.route("/view_tampered")
def view_tampered():
    img = request.args.get("img")
    if not img:
        return "No image specified", 400

    if not os.path.exists(img):
        return "Image not found on server", 404

    output_path = detect_tampered_areas(img)
    
    # Make relative path from "static" folder
    rel_path = os.path.relpath(output_path, "static")
    
    # Convert backslashes to forward slashes for URL
    rel_path = rel_path.replace("\\", "/")
    severity = "Severe"
    return render_template("view_tampered.html", tampered_image=rel_path,severity=severity)

@app.route("/")
def home():
    return render_template("aa.html")
import webbrowser
import threading
chrome_path = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
def open_browser():
    # Register chrome browser controller
    webbrowser.register('chrome', None, webbrowser.BackgroundBrowser(chrome_path))
    webbrowser.get('chrome').open_new("http://127.0.0.1:5000/")

if __name__ == "__main__":
    threading.Timer(1, open_browser).start()
    app.run(debug=True)
