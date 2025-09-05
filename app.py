from dataclasses import dataclass
from typing import Literal
import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# -------------------------
# Data class for message
# -------------------------
@dataclass
class Message:
    origin: Literal["human", "ai"]
    message: str

# -------------------------
# Load CSS
# -------------------------
def load_css():
    with open("static/styles.css", "r") as f:
        css = f"<style>{f.read()}</style>"
        st.markdown(css, unsafe_allow_html=True)

# -------------------------
# Load Model & Vectorizer
# -------------------------
model = joblib.load("intent_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    tokens = nltk.word_tokenize(text)
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words]
    return " ".join(tokens)

def predict_intent(user_input):
    cleaned = clean_text(user_input)
    vec = vectorizer.transform([cleaned])
    tag = model.predict(vec)[0]
    return tag

def get_weather():
    return "☀️ Weather API will be integrated here."

def get_response(intent):
    if intent == "greetings":
        return "Hello, how can I help you?"
    elif intent == "weather_query":
        return get_weather()
    elif intent == "crop_recommendation":
        return "Please provide soil type, rainfall, season..."
    elif intent == "disease_detection":
        return "Upload a crop image 📷 for disease analysis."
    else:
        return "Sorry, I didn’t understand that. Can you rephrase?"

# -------------------------
# Initialize Session
# -------------------------
def initialize_session_state():
    if "history" not in st.session_state:
        st.session_state.history = []

# -------------------------
# Streamlit App
# -------------------------
st.set_page_config(page_title="🌾 Smart Farmer Chatbot", page_icon="🤖", layout="wide")

load_css()
initialize_session_state()

st.title("🌾 Smart Farmer Chatbot 🤖")

chat_placeholder = st.container()

# Display chat history
with chat_placeholder:
    for chat in st.session_state.history:
        div = f"""
<div class="chat-row {'row-reverse' if chat.origin == 'human' else ''}">
    <img class="chat-icon" src="app/static/{'user_icon.png' if chat.origin == 'human' else 'ai_icon.png'}" width=32 height=32>
    <div class="chat-bubble {'human-bubble' if chat.origin == 'human' else 'ai-bubble'}">
        &#8203;{chat.message}
    </div>
</div>
        """
        st.markdown(div, unsafe_allow_html=True)

    for _ in range(2):
        st.markdown("")

# Chat input at bottom
if prompt := st.chat_input("Type your message..."):
    # User message
    st.session_state.history.append(Message("human", prompt))

    # Bot reply
    intent = predict_intent(prompt)
    bot_reply = get_response(intent)
    st.session_state.history.append(Message("ai", bot_reply))

    st.rerun()
