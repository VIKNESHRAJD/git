%%writefile app.py

import streamlit as st
import numpy as np
import librosa
import tensorflow as tf
import torch
from transformers import Wav2Vec2Processor, Wav2Vec2Model
from sklearn.preprocessing import LabelEncoder

# Load trained model
model = tf.keras.models.load_model("aquatic_species_model.h5")

# Define species labels
labels = [
    "AtlanticSpottedDolphin", "CommonDolphin", "SpinnerDolphin", "Commerson'sDolphin", "StripedDolphin",
    "Dall'sPorpoise", "WeddellSeal", "Fin_FinbackWhale", "KillerWhale", "SpottedSeal", "NorthernRightWhale",
    "DuskyDolphin", "BeardedSeal", "HarborPorpoise", "BottlenoseDolphin", "GraySeal", "Short_Finned(Pacific)PilotWhale",
    "SpermWhale", "Narwhal", "Beluga_WhiteWhale", "MinkeWhale", "Fraser'sDolphin", "HarpSeal", "PantropicalSpottedDolphin",
    "RingedSeal", "TucuxiDolphin", "White_sidedDolphin", "White_beakedDolphin", "WestIndianManatee", "Walrus",
    "RossSeal", "SouthernRightWhale", "Boutu_AmazonRiverDolphin", "LeopardSeal", "JuanFernandezFurSeal",
    "SeaOtter", "HarbourSeal", "Long_FinnedPilotWhale", "LongBeaked(Pacific)CommonDolphin", "Grampus_Risso'sDolphin",
    "NewZealandFurSeal", "StellerSeaLion", "BlueWhale", "ClymeneDolphin", "HumpbackWhale", "Rough_ToothedDolphin",
    "GrayWhale", "HoodedSeal", "BowheadWhale", "MelonHeadedWhale", "RibbonSeal", "Heaviside'sDolphin",
    "FinlessPorpoise", "FalseKillerWhale", "IrawaddyDolphin"
]
le = LabelEncoder().fit(labels)

# Load Wav2Vec2.0 model
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
wav2vec_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")

# Feature extraction function
def extract_features(audio_path):
    audio, _ = librosa.load(audio_path, sr=16000)

    # Extract MFCC features
    mfccs = librosa.feature.mfcc(y=audio, sr=16000, n_mfcc=60)
    mfccs_scaled = np.mean(mfccs.T, axis=0)

    # Extract Wav2Vec2.0 features
    inputs = processor(audio, sampling_rate=16000, return_tensors="pt", padding=True)
    with torch.no_grad():
        wav2vec_features = wav2vec_model(**inputs).last_hidden_state.mean(dim=1).squeeze().numpy()

    # Combine features
    return np.hstack((mfccs_scaled, wav2vec_features))

# Streamlit UI
st.title("🐬 Aquatic Species Classifier 🌊")
uploaded_file = st.file_uploader("Upload an audio file (WAV)", type=["wav"])

if uploaded_file:
    st.audio(uploaded_file)
    features = extract_features(uploaded_file)
    features = np.expand_dims(features, axis=(0, 2))
    prediction = model.predict(features)
    predicted_label = labels[np.argmax(prediction)]
    st.subheader(f"🎯 Predicted Species: {predicted_label}")
