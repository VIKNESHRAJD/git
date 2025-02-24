import streamlit as st
import numpy as np
import librosa
import tensorflow as tf
import os

# Load trained model (ensure the correct path)
MODEL_PATH = "aquatic_species_model.h5"

# Check if model file exists
if not os.path.exists(MODEL_PATH):
    st.error(f"Model file not found: {MODEL_PATH}")
    st.stop()

model = tf.keras.models.load_model(MODEL_PATH)

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

# Function to extract MFCC features from the uploaded audio file
def extract_features(file):
    try:
        y, sr = librosa.load(file, sr=22050)  # Load audio with sample rate 22.05 kHz
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)  # Extract 40 MFCC features
        mfccs = np.mean(mfccs, axis=1)  # Take the mean of MFCCs along the time axis
        return mfccs
    except Exception as e:
        st.error(f"Error processing audio file: {e}")
        return None

# Streamlit UI
st.title("🐋 Aquatic Species Classifier")
st.write("Upload a WAV file to predict the aquatic species based on audio features.")

uploaded_file = st.file_uploader("Upload a WAV file", type=["wav"])

if uploaded_file:
    st.audio(uploaded_file)  # Play the uploaded audio
    features = extract_features(uploaded_file)
    
    if features is not None:
        features = np.expand_dims(features, axis=(0, 2))  # Reshape for model input
        prediction = model.predict(features)
        predicted_label = labels[np.argmax(prediction)]
        st.subheader(f"**Predicted Species:** 🐬 {predicted_label}")
    else:
        st.error("Could not extract features from the uploaded file.")
