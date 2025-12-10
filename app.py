import os
import librosa
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import gradio as gr
import shutil
import time

# ======= Helper Functions =======
def extract_features(file_path):
    try:
        y, sr = librosa.load(file_path)
        pitch = librosa.yin(y, fmin=50, fmax=300, sr=sr)
        avg_pitch = np.mean(pitch)
        zcr = np.mean(librosa.feature.zero_crossing_rate(y))
        centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_mean = np.mean(mfcc, axis=1)
        return [avg_pitch, zcr, centroid] + mfcc_mean.tolist()
    except:
        return None

def train_user_model(username, audio_files):
    os.makedirs(f"user_voice_data", exist_ok=True)
    data = []
    for file in audio_files:
        try:
            mood = file.name.split("_")[2].lower()
            file.save(file.name)  # save uploaded file
            features = extract_features(file.name)
            if features:
                data.append([username, mood] + features)
        except:
            continue
    df = pd.DataFrame(data)
    columns = ["user", "mood", "pitch", "zcr", "centroid"] + [f"mfcc{i}" for i in range(1,14)]
    df.columns = columns
    df.to_csv(f"user_voice_data/{username}_profile.csv", index=False)
    return "✅ Training completed!"

def predict_mood(username, test_file, photo_files):
    # Save slideshow photos
    photo_folder = f"user_photos/{username}"
    os.makedirs(photo_folder, exist_ok=True)
    for f in photo_files:
        f.save(os.path.join(photo_folder, f.name))
    
    # Load user's dataset
    try:
        df = pd.read_csv(f"user_voice_data/{username}_profile.csv")
    except:
        return "❌ No trained data for this user.", []
    
    X = df.iloc[:,2:].values
    y = df["mood"].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(X_scaled, y)

    # Save and extract features from test audio
    test_file.save("temp_test.wav")
    X_test = extract_features("temp_test.wav")
    if not X_test:
        return "⚠️ Invalid test audio.", []
    X_test_scaled = scaler.transform([X_test])
    pred = knn.predict(X_test_scaled)[0]

    # Confidence calculation
    nearest_distances, nearest_idx = knn.kneighbors([X_test_scaled[0]], n_neighbors=1)
    confidence = 1 - nearest_distances[0][0] / nearest_distances.max()
    accuracy = min(max(confidence*100,0),100)

    if accuracy < 90:
        return "⚠️ Accuracy < 90%. No mood displayed.", []
    
    if pred.lower() in ["sadness", "anger"]:
        images = [os.path.join(photo_folder, f) for f in os.listdir(photo_folder) if f.lower().endswith((".png",".jpg",".jpeg"))]
        return f"🎯 Predicted mood: {pred.upper()} (Confidence: {accuracy:.2f}%)", images
    else:
        return f"🎯 Predicted mood: {pred.upper()} (Confidence: {accuracy:.2f}%) ❤️", []

# ======= Gradio Interface =======
with gr.Blocks() as demo:
    gr.Markdown("## 🗣️ Multi-User Mood Detection Demo")
    
    with gr.Tab("Train User"):
        username_train = gr.Textbox(label="Enter username")
        audio_train = gr.File(file_types=[".wav"], file_types_accept_multiple=True, label="Upload 6 mood audio files")
        btn_train = gr.Button("Train")
        output_train = gr.Textbox()
        btn_train.click(train_user_model, inputs=[username_train, audio_train], outputs=output_train)
    
    with gr.Tab("Test Mood"):
        username_test = gr.Textbox(label="Enter username")
        test_audio = gr.File(file_types=[".wav"], label="Upload test audio")
        photos = gr.File(file_types=[".png",".jpg",".jpeg"], file_types_accept_multiple=True, label="Upload slideshow images")
        btn_test = gr.Button("Predict Mood")
        output_test = gr.Textbox()
        output_images = gr.Gallery(label="Slideshow Images")
        btn_test.click(predict_mood, inputs=[username_test, test_audio, photos], outputs=[output_test, output_images])

demo.launch()
