import streamlit as st
import numpy as np
import librosa
import tensorflow as tf
import os
from transformers import Wav2Vec2Processor, Wav2Vec2Model
import torch
from sklearn.preprocessing import LabelEncoder

# Load the trained model
model = tf.keras.models.load_model("aquatic_species_model.h5")

# Load Wav2Vec2.0 model
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
wav2vec_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")

# Load label encoder (Make sure to use the same labels as during training)
# Define the list of species labels (Must match training labels)
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

# Encode labels
le = LabelEncoder().fit(labels)


# Function to extract features
def extract_features(audio_path):
    audio, _ = librosa.load(audio_path, sr=16000)
    mfccs = librosa.feature.mfcc(y=audio, sr=16000, n_mfcc=60)
    mfccs_scaled = np.mean(mfccs.T, axis=0)
    
    # Extract Wav2Vec2.0 embeddings
    inputs = processor(audio, sampling_rate=16000, return_tensors="pt", padding=True)
    with torch.no_grad():
        wav2vec_features = wav2vec_model(**inputs).last_hidden_state.mean(dim=1).squeeze().numpy()
    
    # Combine MFCC and Wav2Vec2.0 features
    combined_features = np.hstack((mfccs_scaled, wav2vec_features))
    return combined_features

# Streamlit UI
st.title("Aquatic Species Classification")
st.write("Upload an audio file to classify the species.")

# File uploader
uploaded_file = st.file_uploader("Upload WAV file", type=["wav"])

if uploaded_file is not None:
    st.audio(uploaded_file, format="audio/wav")

    # Save the uploaded file temporarily
    file_path = "temp.wav"
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Extract features
    features = extract_features(file_path)
    features = np.expand_dims(features, axis=0)  # Reshape for model input
    features = np.expand_dims(features, axis=2)  # CNN expects 3D input

    # Make prediction
    prediction = model.predict(features)
    predicted_label = np.argmax(prediction)

    # Display result
    st.subheader("Prediction")
    st.write(f"Predicted Species: **{le.inverse_transform([predicted_label])[0]}**")

    # Cleanup temp file
    os.remove(file_path)
