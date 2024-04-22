import streamlit as st
import librosa
import numpy as np
from keras.models import load_model
import soundfile as sf
import tempfile
import os
import base64
import pyaudio
import wave
import tempfile

# portaudio19-dev
# python3-pyaudio


# Load the trained LSTM model
model = load_model('./model.h5')

# Function to start audio recording
def record_audio(seconds=5, sample_rate=44100, channels=2, chunk_size=1024):
    audio = pyaudio.PyAudio()
    stream = audio.open(format=pyaudio.paInt16,
                        channels=channels,
                        rate=sample_rate,
                        input=True,
                        frames_per_buffer=chunk_size)
    frames = []
    for _ in range(int(sample_rate / chunk_size * seconds)):
        data = stream.read(chunk_size)
        frames.append(data)
    stream.stop_stream()
    stream.close()
    audio.terminate()
    return b''.join(frames)

# Function to save a temp .wav file
def save_wav_file(frames, file_path):
    with wave.open(file_path, 'wb') as wf:
        wf.setnchannels(2)
        wf.setsampwidth(pyaudio.PyAudio().get_sample_size(pyaudio.paInt16))
        wf.setframerate(44100)
        wf.writeframes(frames)

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

# Main function
def main():
    st.title("SER by MAZER")
    st.title("Record now")
    seconds_to_record = st.slider("Record time (seconds)", 0, 10, 5)
    record_button = st.button("Record")

    if record_button:
        st.write("Recording...")
        frames = record_audio(seconds=seconds_to_record+1)
        st.write("Recording complete!")

        # Save the recorded audio to a temporary WAV file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            save_wav_file(frames, tmp_file.name)
            audio_path = tmp_file.name
            st.audio(audio_path, format='audio/wav')
            predicted_emotion = predict_emotion(audio_path, model)
            st.write(f"Predicted Emotion: {predicted_emotion}")

        # Provide download link for the WAV file

if __name__ == "__main__":
    main()



# if we want to upload a file

st.title("OR")

uploaded_file = st.file_uploader("Upload an audio file", type=["wav"])

if uploaded_file is not None:
    st.audio(uploaded_file, format='audio/wav')
    predicted_emotion = predict_emotion(uploaded_file, model)
    st.write(f"Predicted Emotion: {predicted_emotion}")

# should not recognise the voice of a particular person
# real time audio input