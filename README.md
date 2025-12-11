# 🎧 **ManaMood – Personalized Voice Mood Detection**

### *A child project of PulseBond*

ManaMood is an intelligent mood-detection system that identifies a user's emotional state using their **voice tone, pitch, ZCR, spectral features**, and **personalized emotional baseline**.
It listens **only to the original user**, similar to Siri or Google Assistant, and predicts mood through machine learning based on your own voice data.

This project is part of the **PulseBond-When Love Speaks it Listens**.



## 🚀 **Features**

* 🎙️ **Record voice samples for 6 moods**

  * happiness
  * sadness
  * anger
  * calmness
  * excitement
  * romantic

* 🔊 **Browser-based microphone recording** using JavaScript

* 🧠 **Extract audio features**

  * Pitch (F0)
  * Zero Crossing Rate
  * Spectral Centroid

* 🤖 **Train personalized mood model** using KNN

* 🎯 **Predict mood from new audio sample**

* 🖼️ **Show images/slideshow for negative moods (sadness/anger)**

* ❤️ **Show positive emoji for happy moods**

* 🔐 **User-specific emotional profile stored as CSV**



## 📌 **Tech Stack**

* **Python**
* **LibROSA** (audio feature extraction)
* **scikit-learn** (ML model)
* **pandas**
* **JavaScript** (recording UI)
* **Google Colab** (runtime)



## 📂 **Project Structure**

```
ManaMood/
│
├── user_voice_data/               # Auto-generated user voice profiles
├── user_photos/                   # User uploaded images
│
├── main.ipynb                     # All-in-one demo notebook
├── README.md
└── requirements.txt
```



## 🧠 **How ManaMood Works**

ManaMood follows personalized emotional recognition as described in **PulseBond's voice emotion model**:

### 1️⃣ Record Training Samples

User records 6 mood-based voice samples using the built-in UI.

### 2️⃣ Extract Audio Features

For each `.wav` file, ManaMood extracts:

* Average pitch (F0)
* ZCR
* Spectral centroid

### 3️⃣ Create Personalized Emotional Profile

All extracted features are saved in

```
user_voice_data/<username>_profile.csv
```

### 4️⃣ Predict Mood

Using KNN (k=1), ManaMood compares the new audio file with the user’s emotional baseline and predicts the closest mood.

### 5️⃣ Show Emotional Response

* 😔 sadness → show slideshow
* 😡 anger → show slideshow
* 😊 happiness → ❤️
* 😌 calmness → ❤️
* 😍 romantic → ❤️
* 🤩 excitement → ❤️



## 📌 **Future Enhancements**

* Flask backend + Render deployment
* Auto background recording every 30 seconds
* Speaker verification using Resemblyzer
* Android app integration
* SQLite mood history
* Notifications based on mood patterns



## 👨‍💻 **Author**

**G. Bhanu Prakash Reddy**
Creator of **ManaMood** & **PulseBond**-When Love Speaks it Listens.
