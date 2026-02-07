import os
import numpy as np
import librosa
import tensorflow as tf

# --- CONFIGURATION ---
MODEL_PATH = "final_model.h5"
SAMPLE_RATE = 22050
DURATION = 3
N_MELS = 128
MAX_TIME_STEPS = 130
EMOTIONS = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']

def predict_emotion(file_path):
    # 1. Load the Model
    if not os.path.exists(MODEL_PATH):
        print("Error: 'final_model.h5' not found! Did you rename it?")
        return

    model = tf.keras.models.load_model(MODEL_PATH)

    # 2. Process the Audio (Exactly like training)
    audio, sr = librosa.load(file_path, sr=SAMPLE_RATE, duration=DURATION)
    audio, _ = librosa.effects.trim(audio)
    
    # Pad/Crop
    target_length = int(SAMPLE_RATE * DURATION)
    if len(audio) < target_length:
        audio = np.pad(audio, (0, target_length - len(audio)))
    else:
        audio = audio[:target_length]
        
    # Spectrogram
    mels = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=N_MELS)
    log_mels = librosa.power_to_db(mels, ref=np.max)
    
    # Fix Shape
    if log_mels.shape[1] < MAX_TIME_STEPS:
        log_mels = np.pad(log_mels, ((0, 0), (0, MAX_TIME_STEPS - log_mels.shape[1])))
    else:
        log_mels = log_mels[:, :MAX_TIME_STEPS]
    
    # Reshape for AI (Batch, Height, Width, Channels)
    input_data = log_mels.reshape(1, N_MELS, MAX_TIME_STEPS, 1)

    # 3. Predict
    prediction = model.predict(input_data)
    predicted_index = np.argmax(prediction)
    confidence = np.max(prediction) * 100
    emotion = EMOTIONS[predicted_index]
    
    print("\n" + "="*30)
    print(f"File: {os.path.basename(file_path)}")
    print(f"PREDICTION: {emotion.upper()} ({confidence:.1f}%)")
    print("="*30 + "\n")

if __name__ == "__main__":
    # TEST CASE: Pick a file from your data folder to test
    # This path assumes you are in the ai_project folder
    test_file = "./data/Actor_01/03-01-05-01-01-01-01.wav" 
    
    predict_emotion(test_file)