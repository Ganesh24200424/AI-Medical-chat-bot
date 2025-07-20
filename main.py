import streamlit as st
import google.generativeai as genai
import speech_recognition as sr
import base64
import os
import tempfile
from gtts import gTTS
from bs4 import BeautifulSoup
import re

# Set up Google AI API
API_KEY = "AIzaSyCASRzdS93I_fjQaGvCuUW9OQp9YNgnhx8"
genai.configure(api_key=API_KEY)

# Function to list available models
@st.cache_data
def get_available_models():
    try:
        models = genai.list_models()
        return [model.name for model in models]
    except Exception as e:
        st.error(f"Error fetching models: {e}")
        return []

# Function to clean markup from text
def clean_html(text):
    return BeautifulSoup(text, "html.parser").get_text()

# Function to generate AI response
def get_response(prompt, model_name):
    try:
        model = genai.GenerativeModel(model_name)
        response = model.generate_content(prompt)
        cleaned_response = clean_html(response.text) if response else "No response."
        return cleaned_response
    except Exception as e:
        return f"Error: {e}"

# Function to clean text (remove HTML & markdown)
def clean_text(text):
    text = BeautifulSoup(text, "html.parser").get_text()  # Remove HTML
    text = re.sub(r"[*#_~>`]", "", text)  # Remove markdown symbols
    return text.strip()

# Function to generate and store audio file
def generate_audio(text):
    try:
        cleaned_text = clean_text(text)
        if not cleaned_text:
            st.error("Error: The response is empty after cleaning.")
            return None

        tts = gTTS(cleaned_text)
        temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        tts.save(temp_audio.name)
        return temp_audio.name
    except Exception as e:
        st.error(f"Error in text-to-speech: {e}")
        return None

# Function to create HTML audio player with 1.25x speed
def get_audio_html(audio_path):
    try:
        with open(audio_path, "rb") as audio_file:
            audio_base64 = base64.b64encode(audio_file.read()).decode()

        return f"""
        <audio id="responseAudio" controls autoplay>
            <source src="data:audio/mp3;base64,{audio_base64}" type="audio/mpeg">
            Your browser does not support the audio element.
        </audio>
        <script>
            var audio = document.getElementById('responseAudio');
            audio.oncanplay = function() {{
                audio.playbackRate = 1.25;
            }};
        </script>
        """
    except Exception as e:
        st.error(f"Error in generating audio player: {e}")
        return None

# Function to recognize speech
def recognize_speech():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("Listening... Speak now!")
        try:
            audio = recognizer.listen(source, timeout=5)
            return recognizer.recognize_google(audio)
        except sr.UnknownValueError:
            return "Could not understand the audio."
        except sr.RequestError as e:
            return f"Speech recognition request error: {e}"
        except Exception as e:
            return f"Unexpected error: {e}"

# Streamlit UI
st.set_page_config(page_title="AI Voice & Text Chat", layout="centered")

st.title("💡 AI Voice & Text Chat")
st.subheader("Talk to AI in text or voice, and listen to responses!")

# Initialize session state if not set
if "response_text" not in st.session_state:
    st.session_state["response_text"] = None
if "audio_path" not in st.session_state:
    st.session_state["audio_path"] = None

# Fetch available models
models_list = get_available_models()
if not models_list:
    st.error("No models available. Check API key or internet connection.")

# Default model selection (17th index)
default_index = 20 if len(models_list) > 20 else 0
selected_model = st.selectbox("Choose AI Model:", models_list, index=default_index) if models_list else None

if selected_model:
    input_mode = st.radio("Choose Input Mode:", ("Text 📝", "Voice 🎤"), horizontal=True)

    if input_mode == "Text 📝":
        user_input = st.text_area("Enter your message:", height=100)
        if st.button("🚀 Send"):
            if user_input.strip():
                response = get_response(user_input, selected_model)
                st.session_state["response_text"] = response  # Store response
                st.session_state["audio_path"] = None  # Reset audio path to allow new generation
                st.write("🤖 AI Response:", response)

    elif input_mode == "Voice 🎤":
        if st.button("🎙 Record & Send"):
            text = recognize_speech()
            if text:
                st.write("🎤 You said:", text)
                response = get_response(text, selected_model)
                st.session_state["response_text"] = response  # Store response
                st.session_state["audio_path"] = None  # Reset audio path to allow new generation
                st.write("🤖 AI Response:", response)

    # Always display AI response at the top
    if st.session_state["response_text"]:
        st.write("🤖 AI Response:", st.session_state["response_text"])

    # "Listen to Response" button (only generates once)
    if st.session_state["response_text"] and st.button("🎧 Listen to Response"):
        if not st.session_state["audio_path"]:  # Generate only if not already stored
            audio_path = generate_audio(st.session_state["response_text"])
            if audio_path:
                st.session_state["audio_path"] = audio_path

    # Play audio without re-generating
    if st.session_state["audio_path"]:
        audio_html = get_audio_html(st.session_state["audio_path"])
        if audio_html:
            st.markdown(audio_html, unsafe_allow_html=True)

else:
    st.warning("Please select a valid AI model.")