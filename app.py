import os
import pickle
import numpy as np
import librosa
from flask import Flask, render_template, request, jsonify
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings("ignore")

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16MB max upload
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

MODEL_PATH = "model.pkl"
SCALER_PATH = "scaler.pkl"
ENCODER_PATH = "encoder.pkl"
RESULTS_PATH = "results.pkl"

EMOTIONS = {
    "01": "neutral", "02": "calm", "03": "happy", "04": "sad",
    "05": "angry", "06": "fearful", "07": "disgust", "08": "surprised"
}

EMOTION_EMOJI = {
    "neutral": "😐", "calm": "😌", "happy": "😊", "sad": "😢",
    "angry": "😠", "fearful": "😨", "disgust": "🤢", "surprised": "😲"
}


def extract_features(file_path):
    try:
        y, sr = librosa.load(file_path, duration=3, offset=0.5, sr=22050)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        mel = librosa.feature.melspectrogram(y=y, sr=sr)
        zcr = librosa.feature.zero_crossing_rate(y)
        rms = librosa.feature.rms(y=y)

        features = np.hstack([
            np.mean(mfcc, axis=1), np.std(mfcc, axis=1),
            np.mean(chroma, axis=1),
            np.mean(mel, axis=1)[:20],
            np.mean(zcr), np.mean(rms)
        ])
        return features
    except Exception as e:
        print(f"Feature extraction error: {e}")
        return None


def find_dataset():
    # Look for RAVDESS dataset in common locations
    possible_paths = [
        os.path.expanduser("~/Documents/RAVDESS"),
        os.path.expanduser("~/Documents/speech-emotion"),
        os.path.expanduser("~/Downloads/RAVDESS"),
        "dataset/RAVDESS"
    ]
    for path in possible_paths:
        if os.path.exists(path):
            return path
    return None


def train_model():
    dataset_path = find_dataset()
    if dataset_path is None:
        print("RAVDESS dataset not found. Please download it from Kaggle.")
        return None

    X, y = [], []
    for root, dirs, files in os.walk(dataset_path):
        for fname in files:
            if fname.endswith(".wav"):
                parts = fname.replace(".wav", "").split("-")
                if len(parts) >= 3:
                    emotion_code = parts[2]
                    emotion = EMOTIONS.get(emotion_code)
                    if emotion:
                        features = extract_features(os.path.join(root, fname))
                        if features is not None:
                            X.append(features)
                            y.append(emotion)

    if len(X) < 50:
        print(f"Not enough samples found: {len(X)}")
        return None

    X = np.array(X)
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_sc, y_enc, test_size=0.2, random_state=42, stratify=y_enc)

    model = MLPClassifier(
        hidden_layer_sizes=(256, 128, 64),
        activation="relu",
        solver="adam",
        max_iter=500,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = round(accuracy_score(y_test, y_pred) * 100, 2)

    with open(MODEL_PATH, "wb") as f: pickle.dump(model, f)
    with open(SCALER_PATH, "wb") as f: pickle.dump(scaler, f)
    with open(ENCODER_PATH, "wb") as f: pickle.dump(le, f)

    emotion_accs = []
    report = classification_report(y_test, y_pred, target_names=le.classes_, output_dict=True)
    for emo in le.classes_:
        if emo in report:
            emotion_accs.append({
                "emotion": emo,
                "emoji": EMOTION_EMOJI.get(emo, ""),
                "f1": round(report[emo]["f1-score"] * 100, 1)
            })

    all_results = {
        "accuracy": acc,
        "samples": len(X),
        "emotion_scores": emotion_accs,
        "emotions": list(le.classes_)
    }
    with open(RESULTS_PATH, "wb") as f:
        pickle.dump(all_results, f)

    print(f"Training complete. Accuracy: {acc}%")
    return all_results


MODEL = None
SCALER = None
ENCODER = None
RESULTS = None
DATASET_MISSING = False

if os.path.exists(MODEL_PATH) and os.path.exists(RESULTS_PATH):
    print("Loading cached model...")
    with open(MODEL_PATH, "rb") as f: MODEL = pickle.load(f)
    with open(SCALER_PATH, "rb") as f: SCALER = pickle.load(f)
    with open(ENCODER_PATH, "rb") as f: ENCODER = pickle.load(f)
    with open(RESULTS_PATH, "rb") as f: RESULTS = pickle.load(f)
elif find_dataset():
    print("Training model...")
    RESULTS = train_model()
    if RESULTS and os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, "rb") as f: MODEL = pickle.load(f)
        with open(SCALER_PATH, "rb") as f: SCALER = pickle.load(f)
        with open(ENCODER_PATH, "rb") as f: ENCODER = pickle.load(f)
else:
    print("RAVDESS dataset not found.")
    DATASET_MISSING = True
    RESULTS = {
        "accuracy": None, "samples": 0,
        "emotion_scores": [], "emotions": list(EMOTIONS.values())
    }


@app.route("/")
def index():
    return render_template("index.html", results=RESULTS, dataset_missing=DATASET_MISSING)


@app.route("/predict", methods=["POST"])
def predict():
    if MODEL is None:
        return jsonify({"error": "Model not trained. Please download the RAVDESS dataset first."}), 400

    if "audio" not in request.files:
        return jsonify({"error": "No audio file uploaded."}), 400

    f = request.files["audio"]
    if not f.filename.endswith((".wav", ".mp3", ".ogg")):
        return jsonify({"error": "Please upload a .wav, .mp3, or .ogg file."}), 400

    path = os.path.join(UPLOAD_FOLDER, "temp_audio.wav")
    f.save(path)

    features = extract_features(path)
    if features is None:
        return jsonify({"error": "Could not extract features from audio."}), 400

    features_sc = SCALER.transform(features.reshape(1, -1))
    pred_id = MODEL.predict(features_sc)[0]
    proba = MODEL.predict_proba(features_sc)[0]

    emotion = ENCODER.inverse_transform([pred_id])[0]
    confidence = round(float(proba[pred_id]) * 100, 1)

    top3 = sorted(
        [{"emotion": ENCODER.inverse_transform([i])[0], "emoji": EMOTION_EMOJI.get(ENCODER.inverse_transform([i])[0], ""), "prob": round(float(p) * 100, 1)}
         for i, p in enumerate(proba)],
        key=lambda x: x["prob"], reverse=True
    )[:3]

    return jsonify({
        "emotion": emotion,
        "emoji": EMOTION_EMOJI.get(emotion, ""),
        "confidence": confidence,
        "top3": top3
    })


if __name__ == "__main__":
    app.run(debug=False, port=5014)
