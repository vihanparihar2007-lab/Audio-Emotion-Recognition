import os
import numpy as np
import librosa
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, BatchNormalization, MaxPooling2D, GlobalAveragePooling2D, Dense, Dropout
import tensorflow as tf

# --- GPU CHECK (Just to be sure!) ---
print("--------------------------------------------------")
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"SUCCESS! AI is using your: {gpus[0]}")
else:
    print("WARNING: Running on CPU (This will be slower)")
print("--------------------------------------------------")

# --- CONFIGURATION ---
# IMPORTANT: This assumes your folder is named 'data' inside the project folder
DATA_PATH = "./data/" 
SAMPLE_RATE = 22050
DURATION = 3  # Seconds
N_MELS = 128
MAX_TIME_STEPS = 130

def load_data(data_dir):
    features = []
    labels = []
    
    # Map filenames to emotions (RAVDESS codes)
    emotion_map = {
        '01': 'neutral', '02': 'calm', '03': 'happy', '04': 'sad',
        '05': 'angry', '06': 'fearful', '07': 'disgust', '08': 'surprised'
    }

    print("Loading audio files... (This might take 2-3 minutes)")
    
    # Walk through the directory to find files
    # We use os.walk to find files even if they are in sub-folders
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith(".wav"):
                # 1. Get Label from filename (e.g., 03-01-05-...)
                try:
                    parts = file.split('-')
                    emotion_code = parts[2]
                    emotion_label = emotion_map.get(emotion_code)
                except:
                    continue # Skip files that don't match the format

                # 2. Load Audio
                file_path = os.path.join(root, file)
                audio, sr = librosa.load(file_path, sr=SAMPLE_RATE, duration=DURATION)
                
                # 3. Trim Silence (Critical for accuracy)
                audio, _ = librosa.effects.trim(audio)

                # 4. Pad/Crop to ensure 3 seconds fixed length
                target_length = int(SAMPLE_RATE * DURATION)
                if len(audio) < target_length:
                    audio = np.pad(audio, (0, target_length - len(audio)))
                else:
                    audio = audio[:target_length]
                
                # 5. Convert to Spectrogram (The "Image")
                mels = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=N_MELS)
                log_mels = librosa.power_to_db(mels, ref=np.max)
                
                # 6. Fix Shape (Width must be exactly 130)
                if log_mels.shape[1] < MAX_TIME_STEPS:
                    log_mels = np.pad(log_mels, ((0, 0), (0, MAX_TIME_STEPS - log_mels.shape[1])))
                else:
                    log_mels = log_mels[:, :MAX_TIME_STEPS]
                    
                features.append(log_mels)
                labels.append(emotion_label)
            
    return np.array(features), np.array(labels)

# --- EXECUTION ---
if __name__ == "__main__":
    # 1. Load Data
    X, y = load_data(DATA_PATH)
    print(f"Loaded: {X.shape[0]} audio samples")

    # 2. Prepare Data
    lb = LabelEncoder()
    y_encoded = lb.fit_transform(y)
    y_categorical = to_categorical(y_encoded)
    
    # Add 'Channel' dimension for CNN (Height, Width, 1)
    # This turns (1440, 128, 130) into (1440, 128, 130, 1)
    X_cnn = np.expand_dims(X, axis=-1)

    # Stratified Split (80% Train, 10% Val, 10% Test)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_cnn, y_categorical, test_size=0.2, stratify=y_categorical, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
    )

    # 3. Build Model (CNN)
    model = Sequential([
        # Layer 1
        Conv2D(32, (3, 3), activation='relu', input_shape=(N_MELS, MAX_TIME_STEPS, 1)),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.2),
        
        # Layer 2
        Conv2D(64, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.2),
        
        # Layer 3
        Conv2D(128, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.2),
        
        # Output Layer
        GlobalAveragePooling2D(),
        Dense(128, activation='relu'),
        Dense(8, activation='softmax') # 8 Neurons = 8 Emotions
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # 4. Train
    print("Starting training on RTX 3050...")
    history = model.fit(
        X_train, y_train, 
        validation_data=(X_val, y_val), 
        epochs=30, 
        batch_size=32
    )

    # 5. Save & Evaluate
    model.save("final_model.h5")
    print("Model saved as 'final_model.h5'")
    
    loss, acc = model.evaluate(X_test, y_test)
    print(f"Final Test Accuracy: {acc*100:.2f}%")