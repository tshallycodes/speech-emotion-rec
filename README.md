# 🎙️ Speech Emotion Recognition

A deep learning web app that detects human emotion from speech audio. Upload a voice recording and the AI analyses MFCC, chroma, and mel-spectrogram features extracted with librosa — then classifies it into one of 8 emotions using an MLP neural network trained on the RAVDESS dataset.

## Demo

Upload a .wav / .mp3 / .ogg audio file → AI extracts speech features → Emotion detected with top 3 predictions and confidence scores.

## Results

| Metric | Score |
|--------|-------|
| Accuracy | ~70–75% |
| Training Samples | ~1,400 audio files |
| Emotions | 8 classes |
| Dataset | RAVDESS (24 actors, professional recordings) |
| Features | MFCC (40) + Chroma + Mel + ZCR + RMS |

## Emotions Detected

| Emotion | | Emotion | |
|---------|---|---------|---|
| 😐 Neutral | | 😠 Angry | |
| 😌 Calm | | 😨 Fearful | |
| 😊 Happy | | 🤢 Disgust | |
| 😢 Sad | | 😲 Surprised | |

## Features

- Upload .wav, .mp3, or .ogg audio file — instant emotion detection
- Top 3 emotion predictions with confidence percentages
- Per-emotion F1 score chart after training
- Rich feature extraction: MFCC (mean + std), chroma, mel spectrogram, ZCR, RMS
- MLP neural network with 3 hidden layers (256 → 128 → 64)
- Model cached after first training run — instant on second load

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Feature Extraction | librosa (MFCC, Chroma, Mel Spectrogram, ZCR, RMS) |
| Model | MLPClassifier — scikit-learn |
| Preprocessing | StandardScaler, LabelEncoder |
| Web Framework | Flask |
| Dataset | RAVDESS — 1,440 audio files, 24 actors, 8 emotions |
| Language | Python |

## How to Run

**1. Download the RAVDESS dataset**

Download from [Kaggle](https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio) and extract to:
```
~/Documents/RAVDESS/
```

**2. Clone the repo**
```bash
git clone https://github.com/manny2341/speech-emotion-recognition.git
cd speech-emotion-recognition
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Start the app**
```bash
python3 app.py
```
The model trains automatically on first run (~1–2 minutes).

**5. Open in browser**
```
http://127.0.0.1:5014
```

## How It Works

1. Audio file loaded with **librosa** (first 3 seconds, 22050Hz sample rate)
2. Feature extraction:
   - **40 MFCC coefficients** (mean + std = 80 values) — captures timbre and tone
   - **12 Chroma features** — captures harmonic content
   - **20 Mel spectrogram** values — captures frequency energy
   - **ZCR** (zero-crossing rate) and **RMS energy**
3. All features scaled with **StandardScaler**
4. **MLP classifier** predicts emotion from the 100+ dimensional feature vector
5. Top 3 predictions returned with confidence percentages

## Model Architecture

```
Input: ~114 audio features
  → Dense(256, relu)
  → Dense(128, relu)
  → Dense(64, relu)
  → Dense(8, softmax)
Output: 8 emotion probabilities
```

## Project Structure

```
speech-emotion-recognition/
├── app.py               # Flask server, librosa feature extraction, MLP training
├── uploads/             # Temporary audio file storage (auto-created)
├── templates/
│   └── index.html       # Upload UI, emotion result, per-class F1 chart
├── static/
│   └── style.css        # Dark theme styling
└── requirements.txt
```

## My Other ML Projects

| Project | Description | Repo |
|---------|-------------|------|
| Face Recognition Login | OpenCV LBPH — webcam face authentication | [face-recognition-login](https://github.com/manny2341/face-recognition-login) |
| Emotion Detection | CNN — real-time webcam emotion recognition | [Emotion-Detection](https://github.com/manny2341/Emotion-Detection) |
| Fake News Detector | NLP — TF-IDF fake vs real news | [fake-news-detector](https://github.com/manny2341/fake-news-detector) |
| Crop Disease Detector | EfficientNetV2 — 15 plant diseases | [crop-disease-detector](https://github.com/manny2341/crop-disease-detector) |

## Author

[@manny2341](https://github.com/manny2341)
