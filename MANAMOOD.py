# Installing Required Python Packages
!pip install librosa soundfile scikit-learn pandas

# Importing HTML for Browser-Based Recording UI
from IPython.display import HTML

# Function to Show Start/Stop Recording Buttons
def record_ui_fixed(mood):
  display(HTML(f"""
  <script>
  let recorder_{mood}, stream_{mood};
  async function startRecording_{mood}() {{
    stream_{mood} = await navigator.mediaDevices.getUserMedia({{ audio: true }});
    recorder_{mood} = new MediaRecorder(stream_{mood});
    let audioChunks = [];
    recorder_{mood}.ondataavailable = e => audioChunks.push(e.data);
    recorder_{mood}.onstop = e => {{
      let blob = new Blob(audioChunks);
      let url = URL.createObjectURL(blob);
      let a = document.createElement('a');
      a.href = url;
      a.download = '{mood}.wav';
      a.click();
    }};
    recorder_{mood}.start();
  }}
  function stopRecording_{mood}() {{
    recorder_{mood}.stop();
    stream_{mood}.getTracks().forEach(track => track.stop());
  }}
  </script>
  <div>
    <h4>🎤 Record for: {mood.upper()}</h4>
    <button onclick="startRecording_{mood}()">Start Recording</button>
    <button onclick="stopRecording_{mood}()">Stop & Download</button>
  </div>
  """))

# Creating a Recorder UI for All 6 Moods
moods = ["sadness", "happiness", "anger", "calmness", "excitement", "romantic"]
for mood in moods:
  record_ui_fixed(mood)

from google.colab import files
import os
import shutil

# Create a folder to store uploaded images
os.makedirs('user_photos', exist_ok=True)
print()
print("Upload your favourite Images🖼️")
# Uploading User Images
uploaded = files.upload()

# Move images to user_photos/
for fname in uploaded:
    shutil.move(fname, os.path.join("user_photos", fname))
print(f"Uploaded {len(uploaded)} photo(s).")

from google.colab import files

print()
# Uploading All Mood Audio Files
print("📁 Upload all 6 .wav files (one per mood)")
uploaded = files.upload()

import os
import librosa
import numpy as np
import pandas as pd

def extract_features(file_path):

# Extracting Audio Features
    y, sr = librosa.load(file_path)
    pitch = librosa.yin(y, fmin=50, fmax=300, sr=sr)
    avg_pitch = np.mean(pitch)
    zcr = np.mean(librosa.feature.zero_crossing_rate(y))
    centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    return [avg_pitch, zcr, centroid]

# Asking Username & Creating CSV
username = input("Enter your username: ").strip()
os.makedirs("user_voice_data", exist_ok=True)
data = []
for file_name in uploaded.keys():
    mood = file_name.replace(".wav", "")
    features = extract_features(file_name)
    data.append([username, mood] + features)
df = pd.DataFrame(data, columns=["user", "mood", "pitch", "zcr", "centroid"])
df.to_csv(f"user_voice_data/{username}_profile.csv", index=False)
print("✅ Voice profile saved!")

# Record Test Audio
record_ui_fixed("test_mood")
test = files.upload()

from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

# Predicting Mood Using KNN
test_file = list(test.keys())[0]
X_test = [extract_features(test_file)]

# Load user's dataset
df = pd.read_csv(f"user_voice_data/{username}_profile.csv")
X = df[["pitch", "zcr", "centroid"]]
y = df["mood"]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test)
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_scaled, y)
pred = knn.predict(X_test_scaled)[0]
print(f"\n🎯 Predicted Mood: {pred}")

# Clean the prediction string
mood = pred.strip().lower()

# Mood-based Display
if mood in ["sadness", "anger"]:
    print(f"\nDetected mood is {mood.upper()} — showing slideshow.")
    show_slideshow("user_photos")
else:
    print(f"\nDetected mood is {mood.upper()} — showing love emoji.")
    print("❤️")
