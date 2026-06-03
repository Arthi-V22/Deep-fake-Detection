# AI MediaGuard - Deepfake Audio Detector (CNN + LSTM)
import os
import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from sklearn.metrics import confusion_matrix

#  PARAMETERS 
DATA_DIR = "./AUDIO"        # Must have REAL/FAKE folders
SR = 16000                  # Sampling rate
DURATION = 3                # Audio duration in seconds
MAX_LEN = 130               # Maximum frames for features
N_MFCC = 20                 # Number of MFCC features

#  FEATURE EXTRACTION 
def extract_features(audio, sr=SR, max_len=MAX_LEN, n_mfcc=N_MFCC):
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
    mel = librosa.feature.melspectrogram(y=audio, sr=sr)
    mel_db = librosa.power_to_db(mel)
    features = np.vstack([mfcc, chroma, mel_db])

    if features.shape[1] < max_len:
        pad_width = max_len - features.shape[1]
        features = np.pad(features, ((0,0),(0,pad_width)), mode='constant')
    elif features.shape[1] > max_len:
        features = features[:, :max_len]

    features = (features - np.mean(features)) / (np.std(features) + 1e-9)
    return features

def augment_audio(audio, sr):
    augmented = [audio]  # original

    # Pitch shift
    for n_steps in [-2, -1, 1, 2]:
        shifted = librosa.effects.pitch_shift(y=audio, sr=sr, n_steps=n_steps)
        augmented.append(shifted)
    
    # Time stretch
    for rate in [0.9, 1.05, 1.1]:
        try:
            stretched = librosa.effects.time_stretch(y=audio, rate=rate)
            augmented.append(stretched)
        except Exception as e:
            print(f"[WARN] time_stretch failed: {e}")
    
    # Add noise
    noise = audio + 0.005 * np.random.randn(len(audio))
    augmented.append(noise)

    return augmented

#  LOAD DATA 
def load_dataset(data_dir):
    features, labels = [], []

    # REAL audio (with augmentation)
    real_path = os.path.join(data_dir, "REAL")
    for file in os.listdir(real_path):
        if not file.lower().endswith(".wav"): 
            continue
        audio, sr = librosa.load(os.path.join(real_path, file), sr=SR, duration=DURATION)
        for aug in augment_audio(audio, sr):
            features.append(extract_features(aug))
            labels.append(0)  # real

    # FAKE audio (with augmentation)
    fake_path = os.path.join(data_dir, "FAKE")
    for file in os.listdir(fake_path):
        if not file.lower().endswith(".wav"): 
            continue
        audio, sr = librosa.load(os.path.join(fake_path, file), sr=SR, duration=DURATION)
        for aug in augment_audio(audio, sr):
            features.append(extract_features(aug))
            labels.append(1)  # fake

    return np.array(features), np.array(labels)

#  PREPARE DATA 
print("[INFO] Loading dataset...")
X, y = load_dataset(DATA_DIR)
X = X[..., np.newaxis]  # add channel dimension for CNN

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

cw = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weights = dict(enumerate(cw))
print("[INFO] Class weights:", class_weights)

#  BUILD FIXED CNN+LSTM MODEL 
def build_model(input_shape):
    inp = layers.Input(shape=input_shape)

    # CNN feature extractor
    x = layers.Conv2D(32, (3,3), activation='relu', padding='same')(inp)
    x = layers.MaxPooling2D((2,2))(x)
    x = layers.Conv2D(64, (3,3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2,2))(x)
    x = layers.Conv2D(128, (3,3), activation='relu', padding='same')(x)

    # Keep time dimension for LSTM
    # x.shape = (batch, freq, time, channels)
    freq, time, channels = x.shape[1], x.shape[2], x.shape[3]
    x = layers.Permute((2,1,3))(x)               # swap freq & time → (batch, time, freq, channels)
    x = layers.Reshape((time, freq*channels))(x)  # flatten freq+channels per frame

    # LSTM over temporal dimension
    x = layers.LSTM(64)(x)
    x = layers.Dense(64, activation='relu')(x)
    out = layers.Dense(1, activation='sigmoid')(x)

    model = models.Model(inputs=inp, outputs=out)
    return model

model = build_model(X_train.shape[1:])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

#  TRAIN 
history = model.fit(
    X_train, y_train,
    epochs=25,
    batch_size=16,
    validation_split=0.2,
    class_weight=class_weights,
    verbose=1
)

#  EVALUATE 
loss, acc = model.evaluate(X_test, y_test)
print(f"\n✅ Test Accuracy: {acc*100:.2f}%")

y_pred = (model.predict(X_test) > 0.5).astype(int)
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)
print(f"Real correct: {sum(y_pred[y_test==0]==0)}, Fake correct: {sum(y_pred[y_test==1]==1)}")

#  SAVE MODEL 
os.makedirs("models", exist_ok=True)
model.save("models/fake_audio_detector_report.h5")
print("[INFO] Model saved: models/fake_audio_detector_report.h5")

#  PREDICTION FUNCTION 
def predict_audio(file_path):
    audio, sr = librosa.load(file_path, sr=SR, duration=DURATION)
    features = extract_features(audio)
    features = features[np.newaxis, ..., np.newaxis]
    pred = model.predict(features)
    return "Fake Audio" if pred[0][0] > 0.5 else "Real Audio"

#  DEMO 
print(predict_audio("./AUDIO/REAL/linus-original.wav"))
print(predict_audio("./AUDIO/FAKE/biden-to-Obama.wav"))
