import streamlit as st
import librosa
import numpy as np
from keras.models import load_model

# Load the trained LSTM model
model = load_model('./model.h5')

# Function to extract MFCC features from audio file
def extract_mfcc(filename):
    y, sr = librosa.load(filename, duration=3, offset=0.5)
    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
    return mfcc

# Function to predict emotion from audio file
def predict_emotion(filename, model):
    mfcc_features = extract_mfcc(filename)
    mfcc_features = np.expand_dims(mfcc_features, axis=0)
    mfcc_features = np.expand_dims(mfcc_features, axis=-1)
    prediction = model.predict(mfcc_features)
    predicted_label = np.argmax(prediction)
    emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
    predicted_emotion = emotions[predicted_label]
    return predicted_emotion

# Streamlit app
st.title('Speech Emotion Recognition')

# File uploader for test audio input
uploaded_file = st.file_uploader("Upload an audio file", type=["wav"])

if uploaded_file is not None:
    st.audio(uploaded_file, format='audio/wav')
    predicted_emotion = predict_emotion(uploaded_file, model)
    st.write(f"Predicted Emotion: {predicted_emotion}")
