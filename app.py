from dataclasses import dataclass
from typing import Literal
import streamlit as st
import joblib
import tensorflow as tf
import numpy as np
import re
import os
import tempfile
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pickle
import pandas as pd
import requests
from datetime import datetime, timedelta
import json
import toml
import streamlit.components.v1 as components
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from deep_translator import GoogleTranslator
from langdetect import detect

import nltk
NLTK_DIR = os.path.join(os.getcwd(), "nltk_data")
nltk.data.path.append(NLTK_DIR)

nltk_packages = ["stopwords", "punkt", "wordnet", "omw-1.4"]
for pkg in nltk_packages:
    try:
        nltk.data.find(f"{'tokenizers' if pkg=='punkt' else 'corpora'}/{pkg}")
    except LookupError:
        nltk.download(pkg, download_dir=NLTK_DIR, quiet=True)

# -------------------------
# Data class for message
# -------------------------
@dataclass
class Message:
    origin: Literal["human", "ai"]
    message: str


# Initialize translator
translator = GoogleTranslator(source='auto', target='en')

# Supported languages for the app
SUPPORTED_LANGUAGES = {
    'en': 'English',
    'hi': 'हिंदी (Hindi)',
    'bn': 'বাংলা (Bengali)',
    'te': 'తెలుగు (Telugu)',
    'ta': 'தமிழ் (Tamil)',
    'mr': 'मराठी (Marathi)',
    'gu': 'ગુજરાતી (Gujarati)',
    'kn': 'ಕನ್ನಡ (Kannada)',
    'ml': 'മലയാളം (Malayalam)',
    'pa': 'ਪੰਜਾਬੀ (Punjabi)',
    'or': 'ଓଡ଼ିଆ (Odia)',
    'as': 'অসমীয়া (Assamese)'
}

# Translation functions
def detect_language(text):
    """Detect the language of input text"""
    try:
        detected = detect(text)
        return detected if detected in SUPPORTED_LANGUAGES else 'en'
    except:
        return 'en'

def translate_text(text, target_lang='en', source_lang='auto'):
    """Translate text to target language using deep-translator"""
    try:
        if target_lang == 'en' and detect_language(text) == 'en':
            return text
        
        if source_lang == 'auto':
            source_lang = detect_language(text)
        
        if source_lang == target_lang:
            return text
        
        translator = GoogleTranslator(source=source_lang, target=target_lang)
        result = translator.translate(text)
        return result
    except Exception as e:
        st.error(f"Translation error: {e}")
        return text

def get_ui_text(key, lang='en'):
    """Get UI text in the specified language"""
    ui_texts = {
        'en': {
            'app_title': 'KrishiMitra',
            'app_subtitle': 'Your Smart Farming Assistant',
            'language_selector': 'Select Language',
            'type_message': 'Type your message...',
            'location_prompt': 'Provide your location',
            'submit': 'Submit',
            'location': 'Location',
            'select_state': 'Select State:',
            'weather_report': 'Weather Report for',
            'current_conditions': 'Current Conditions:',
            'temperature': 'Temperature',
            'condition': 'Condition',
            'humidity': 'Humidity',
            'wind_speed': 'Wind Speed',
            'atmospheric_pressure': 'Atmospheric Pressure',
            'daylight_hours': 'Daylight Hours',
            'rainfall': 'Rainfall',
            'agricultural_alerts': 'Agricultural Alerts',
            'farming_recommendations': 'Farming Recommendations',
            'crop_recommendation': 'Crop Recommendation for',
            'soil_profile': 'Soil Profile:',
            'weather_conditions': 'Weather Conditions:',
            'recommended_crop': 'Recommended Crop:',
            'fertilizer_form_title': 'Provide location and crop information for fertilizer recommendation',
            'crop_type': 'Crop Type:',
            'soil_type': 'Soil Type:',
            'get_fertilizer_rec': 'Get Fertilizer Recommendation',
            'fertilizer_rec_report': 'Fertilizer Recommendation Report',
            'environmental_conditions': 'Environmental Conditions:',
            'soil_nutrient_profile': 'Soil Nutrient Profile:',
            'crop_soil_info': 'Crop & Soil Information:',
            'fertilizer_recommendation': 'Fertilizer Recommendation:',
            'application_guidelines': 'Application Guidelines:',
            'important_notes': 'Important Notes:',
            'market_price_form': 'Provide market information for price prediction',
            'enter_date': 'Enter date (DD-MM-YYYY):',
            'enter_district': 'Enter District:',
            'enter_commodity': 'Enter Crop/Commodity:',
            'get_price_prediction': 'Get Price Prediction',
            'market_price_prediction': 'Market Price Prediction',
            'predicted_modal_price': 'Predicted Modal Price',
            'disease_detection_prompt': 'Upload an image of the crop to detect disease',
            'choose_image': 'Choose an image...',
            'crop_disease_result': 'Crop Disease Detection Result',
            'detected_disease': 'Detected Disease:',
            'confidence_level': 'Confidence Level:',
            'recommendation': 'Recommendation:',
            'soil_health_form': 'Provide soil health information',
            'soil_ph': 'Soil pH:',
            'organic_matter': 'Organic Matter (%):',
            'soil_moisture': 'Soil Moisture (%):',
            'get_soil_analysis': 'Get Soil Health Analysis',
            'soil_health_report': 'Soil Health Analysis Report',
            'soil_parameters': 'Soil Parameters:',
            'analysis': 'Analysis:',
            'recommendations': 'Recommendations:',
            'greeting_message': 'Namaste!\n\nI am **KrishiMitra**, How can I assist you today?',
            'weather_location_request': 'Please provide your location.',
            'crop_location_request': 'Please provide your location.',
            'disease_image_request': 'Please upload an image of the crop to detect diseases.',
            'fertilizer_info_request': 'Please provide soil and crop information for fertilizer recommendation.',
            'soil_health_request': 'Please provide soil health parameters for analysis.',
            'market_info_request': 'Please provide market information for price prediction.',
            'fallback_response': 'Sorry, I don\'t have this information. Please contact an agriculture expert.'
        },
        'hi': {
            'app_title': 'कृषिमित्र',
            'app_subtitle': 'आपका स्मार्ट कृषि सहायक',
            'language_selector': 'भाषा चुनें',
            'type_message': 'अपना संदेश टाइप करें...',
            'location_prompt': 'अपना स्थान प्रदान करें',
            'submit': 'जमा करें',
            'location': 'स्थान',
            'select_state': 'राज्य चुनें:',
            'weather_report': 'मौसम रिपोर्ट',
            'current_conditions': 'वर्तमान स्थितियां:',
            'temperature': 'तापमान',
            'condition': 'स्थिति',
            'humidity': 'नमी',
            'wind_speed': 'हवा की गति',
            'atmospheric_pressure': 'वायुमंडलीय दबाव',
            'daylight_hours': 'दिन के प्रकाश के घंटे',
            'rainfall': 'बारिश',
            'agricultural_alerts': 'कृषि अलर्ट',
            'farming_recommendations': 'कृषि सिफारिशें',
            'crop_recommendation': 'फसल सिफारिश',
            'soil_profile': 'मिट्टी प्रोफाइल:',
            'weather_conditions': 'मौसम की स्थिति:',
            'recommended_crop': 'अनुशंसित फसल:',
            'fertilizer_form_title': 'उर्वरक सिफारिश के लिए स्थान और फसल की जानकारी प्रदान करें',
            'crop_type': 'फसल का प्रकार:',
            'soil_type': 'मिट्टी का प्रकार:',
            'get_fertilizer_rec': 'उर्वरक सिफारिश प्राप्त करें',
            'fertilizer_rec_report': 'उर्वरक सिफारिश रिपोर्ट',
            'environmental_conditions': 'पर्यावरणीय स्थितियां:',
            'soil_nutrient_profile': 'मिट्टी पोषक तत्व प्रोफाइल:',
            'crop_soil_info': 'फसल और मिट्टी की जानकारी:',
            'fertilizer_recommendation': 'उर्वरक सिफारिश:',
            'application_guidelines': 'अनुप्रयोग दिशानिर्देश:',
            'important_notes': 'महत्वपूर्ण नोट्स:',
            'market_price_form': 'मूल्य भविष्यवाणी के लिए बाजार की जानकारी प्रदान करें',
            'enter_date': 'दिनांक दर्ज करें (DD-MM-YYYY):',
            'enter_district': 'जिला दर्ज करें:',
            'enter_commodity': 'फसल/कमोडिटी दर्ज करें:',
            'get_price_prediction': 'मूल्य भविष्यवाणी प्राप्त करें',
            'market_price_prediction': 'बाजार मूल्य भविष्यवाणी',
            'predicted_modal_price': 'अनुमानित मॉडल मूल्य',
            'disease_detection_prompt': 'रोग का पता लगाने के लिए फसल की तस्वीर अपलोड करें',
            'choose_image': 'एक तस्वीर चुनें...',
            'crop_disease_result': 'फसल रोग का पता लगाने का परिणाम',
            'detected_disease': 'खोजा गया रोग:',
            'confidence_level': 'विश्वसनीयता स्तर:',
            'recommendation': 'सिफारिश:',
            'soil_health_form': 'मिट्टी स्वास्थ्य की जानकारी प्रदान करें',
            'soil_ph': 'मिट्टी pH:',
            'organic_matter': 'जैविक पदार्थ (%):',
            'soil_moisture': 'मिट्टी की नमी (%):',
            'get_soil_analysis': 'मिट्टी स्वास्थ्य विश्लेषण प्राप्त करें',
            'soil_health_report': 'मिट्टी स्वास्थ्य विश्लेषण रिपोर्ट',
            'soil_parameters': 'मिट्टी पैरामीटर:',
            'analysis': 'विश्लेषण:',
            'recommendations': 'सिफारिशें:',
            'greeting_message': 'नमस्ते!\n\nमैं **कृषिमित्र** हूं, आज मैं आपकी कैसे सहायता कर सकता हूं?',
            'weather_location_request': 'कृपया अपना स्थान प्रदान करें।',
            'crop_location_request': 'कृपया अपना स्थान प्रदान करें।',
            'disease_image_request': 'कृपया रोगों का पता लगाने के लिए फसल की एक तस्वीर अपलोड करें।',
            'fertilizer_info_request': 'उर्वरक सिफारिश के लिए कृपया मिट्टी और फसल की जानकारी प्रदान करें।',
            'soil_health_request': 'विश्लेषण के लिए कृपया मिट्टी स्वास्थ्य पैरामीटर प्रदान करें।',
            'market_info_request': 'मूल्य भविष्यवाणी के लिए कृपया बाजार की जानकारी प्रदान करें।',
            'fallback_response': 'खुशी है, मेरे पास यह जानकारी नहीं है। कृपया एक कृषि विशेषज्ञ से संपर्क करें।'
        },
        'bn': {
            'app_title': 'কৃষিমিত্র',
            'app_subtitle': 'আপনার স্মার্ট কৃষি সহায়ক',
            'language_selector': 'ভাষা নির্বাচন করুন',
            'type_message': 'আপনার বার্তা টাইপ করুন...',
            'location_prompt': 'আপনার অবস্থান প্রদান করুন',
            'submit': 'জমা দিন',
            'location': 'অবস্থান',
            'select_state': 'রাজ্য নির্বাচন করুন:',
            'greeting_message': 'নমস্কার!\n\nআমি **কৃষিমিত্র**, আজ আমি আপনাকে কীভাবে সাহায্য করতে পারি?',
            'weather_location_request': 'দয়া করে আপনার অবস্থান প্রদান করুন।',
            'crop_location_request': 'দয়া করে আপনার অবস্থান প্রদান করুন।',
            'disease_image_request': 'রোগ শনাক্ত করতে দয়া করে ফসলের একটি ছবি আপলোড করুন।',
            'fertilizer_info_request': 'সার সুপারিশের জন্য দয়া করে মাটি এবং ফসলের তথ্য প্রদান করুন।',
            'soil_health_request': 'বিশ্লেষণের জন্য দয়া করে মাটির স্বাস্থ্য পরামিতি প্রদান করুন।',
            'market_info_request': 'দাম পূর্বাভাসের জন্য দয়া করে বাজারের তথ্য প্রদান করুন।',
            'fallback_response': 'দুঃখিত, আমার কাছে এই তথ্য নেই। দয়া করে একজন কৃষি বিশেষজ্ঞের সাথে যোগাযোগ করুন।'
        }
        # Add more languages as needed
    }
    
    return ui_texts.get(lang, ui_texts['en']).get(key, ui_texts['en'][key])

# Initialize session state for language
def initialize_language_session():
    if "selected_language" not in st.session_state:
        st.session_state.selected_language = 'en'
    if "user_input_language" not in st.session_state:
        st.session_state.user_input_language = 'en'

col1, col2 = st.columns([1,10])

with col1:
    st.image("static/logo.png", width=120)  # local file works here

with col2:
    # Initialize language session first
    initialize_language_session()
    
    # Get current language
    current_lang = st.session_state.selected_language
    
    st.markdown(
        f"""
        <h1 style="margin: 0; line-height: 1.2;">{get_ui_text('app_title', current_lang)}</h1>
        <p style="margin: 0; line-height: 0.00000000001;"><i>{get_ui_text('app_subtitle', current_lang)}</i></p>
        """,
        unsafe_allow_html=True
    )

# Language selector in sidebar
with st.sidebar:
    st.image("logo with text.png", width=150)
    st.markdown(f"### {get_ui_text('language_selector', current_lang)}")
    selected_language = st.selectbox(
        "Select language",  # <- give non-empty
        options=list(SUPPORTED_LANGUAGES.keys()),
        format_func=lambda x: SUPPORTED_LANGUAGES[x],
        index=list(SUPPORTED_LANGUAGES.keys()).index(st.session_state.selected_language),
        key="language_selector",
        label_visibility="collapsed"  # hides it
    )
    
    if selected_language != st.session_state.selected_language:
        st.session_state.selected_language = selected_language
        # Translate the greeting message
        greeting = get_ui_text('greeting_message', selected_language)
        if st.session_state.get("history"):
            # Update the first message (greeting) to new language
            st.session_state.history[0] = Message("ai", greeting)
        st.rerun()


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
cb_model = joblib.load("intent_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

# Import crop recommendation models
encoder = pickle.load(open("cr_rec_models/encoder.pkl", 'rb'))
scaler = pickle.load(open("cr_rec_models/scaler.pkl", 'rb'))
model_gbc = pickle.load(open("cr_rec_models/model_gbc.pkl", 'rb'))
soil_df = pd.read_csv("shc_scaled_to_crop_range.csv")

model_url = "https://github.com/ArnabGupta2004/KrishiMitra/releases/download/v1.0/cr_price.pkl"
model_path = "cr_price_models/cr_price.pkl"

def download_file(url, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.exists(path):
        print(f"Downloading {os.path.basename(path)}...")
        r = requests.get(url, stream=True)
        r.raise_for_status()
        with open(path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"{os.path.basename(path)} downloaded successfully!")

# Ensure model file is available
download_file(model_url, model_path)

# Load price prediction model + encoders
model_cr_pr = joblib.load(model_path)
le_commodity = joblib.load("cr_price_models/le_commodity.pkl")
le_district = joblib.load("cr_price_models/le_district.pkl")
le_market = joblib.load("cr_price_models/le_market.pkl")
le_state = joblib.load("cr_price_models/le_state.pkl")
price_df = pd.read_csv("agmarknet_prices.csv")

# Import fertilizer recommendation models
svm_pipeline = pickle.load(open("fertilizer models/svm_pipeline.pkl", 'rb'))
rf_pipeline = pickle.load(open("fertilizer models/rf_pipeline.pkl", 'rb'))
xgb_pipeline = pickle.load(open("fertilizer models/xgb_pipeline.pkl", 'rb'))
fertname_dict = pickle.load(open("fertilizer models/fertname_dict.pkl", 'rb'))
croptype_dict = pickle.load(open("fertilizer models/croptype_dict.pkl", 'rb'))
soiltype_dict = pickle.load(open("fertilizer models/soiltype_dict.pkl", 'rb'))

secrets = toml.load(".streamlit/secrets.toml")
open_weather_api = secrets["openweather"]["api_key"]

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    tokens = nltk.word_tokenize(text)
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words]
    return " ".join(tokens)

def predict_intent(user_input):
    # Translate input to English for intent prediction
    english_input = translate_text(user_input, target_lang='en', source_lang='auto')
    cleaned = clean_text(english_input)
    vec = vectorizer.transform([cleaned])
    tag = cb_model.predict(vec)[0]
    return tag


def crop_model_prediction(uploaded_file):
    try:
        import tempfile
        # Save uploaded image temporarily
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(uploaded_file.read() if hasattr(uploaded_file, 'read') else uploaded_file)
            tmp_file_path = tmp_file.name

        # Load image and resize
        img = image.load_img(tmp_file_path, target_size=(128, 128))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = x / 255.0

        # Load model
        model = tf.keras.models.load_model("trained_plant_disease_model.h5")
        predictions = model.predict(x)

        if predictions.size == 0:
            return "❌ Error: Model returned empty predictions.", None

        predicted_index = np.argmax(predictions[0])
        confidence = predictions[0][predicted_index] * 100

        # Full list of disease classes
        disease_classes = [
            "Apple___Apple_scab",
            "Apple___Black_rot",
            "Apple___Cedar_apple_rust",
            "Apple___healthy",
            "Blueberry___healthy",
            "Cherry_(including_sour)___Powdery_mildew",
            "Cherry_(including_sour)___healthy",
            "Corn_(maize)___Cercospora_leaf_spot_Gray_leaf_spot",
            "Corn_(maize)___Common_rust_",
            "Corn_(maize)___Northern_Leaf_Blight",
            "Corn_(maize)___healthy",
            "Grape___Black_rot",
            "Grape___Esca_(Black_Measles)",
            "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
            "Grape___healthy",
            "Orange___Haunglongbing_(Citrus_greening)",
            "Orange___healthy",
            "Peach___Bacterial_spot",
            "Peach___healthy",
            "Pepper,_bell___Bacterial_spot",
            "Pepper,_bell___healthy",
            "Potato___Early_blight",
            "Potato___Late_blight",
            "Potato___healthy",
            "Raspberry___healthy",
            "Soybean___healthy",
            "Squash___Powdery_mildew",
            "Strawberry___Leaf_scorch",
            "Strawberry___healthy",
            "Tomato___Bacterial_spot",
            "Tomato___Early_blight",
            "Tomato___Late_blight",
            "Tomato___Leaf_Mold",
            "Tomato___Septoria_leaf_spot",
            "Tomato___Spider_mites Two-spotted_spider_mite",
            "Tomato___Target_Spot",
            "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
            "Tomato___Tomato_mosaic_virus",
            "Tomato___healthy"
        ]

        return disease_classes[predicted_index], confidence

    except Exception as e:
        return f"❌ Error during disease prediction: {str(e)}", None

# -------------------------
# Robust Scrolling Helpers
# -------------------------
def scroll_to_bottom():
    """
    Scroll the parent Streamlit page to the bottom (robust with retries and fallbacks).
    Call this after chat is rendered and you want the view to move to the latest messages.
    """
    js = """
    <script>
    (function(){
        var attempts = 0;
        var maxAttempts = 20;
        var interval = 120;
        var id = setInterval(function(){
            attempts++;
            try {
                // Prefer main tag
                var main = window.parent.document.querySelector('main');
                if (main) {
                    main.scrollTo({ top: main.scrollHeight, behavior: 'smooth' });
                    clearInterval(id);
                    return;
                }
                // Fallbacks
                var container = window.parent.document.querySelector('.block-container, .stApp, .main');
                if (container) {
                    container.scrollTo({ top: container.scrollHeight, behavior: 'smooth' });
                    clearInterval(id);
                    return;
                }
                // Ultimate fallback: window
                window.parent.scrollTo({ top: window.parent.document.body.scrollHeight, behavior: 'smooth' });
                clearInterval(id);
            } catch (e) {
                // ignore and retry
            }
            if (attempts >= maxAttempts) clearInterval(id);
        }, interval);
    })();
    </script>
    """
    components.html(js, height=0)

def scroll_to_anchor(anchor_id: str):
    """
    Scroll parent document to the element with id=anchor_id (retries until found).
    Useful for scrolling to dynamic input widgets.
    """
    js = f"""
    <script>
    (function(){{
        var attempts = 0;
        var maxAttempts = 25;
        var interval = 120;
        var id = setInterval(function(){{
            attempts++;
            try {{
                var anchor = window.parent.document.getElementById('{anchor_id}');
                if (anchor) {{
                    anchor.scrollIntoView({{behavior: 'smooth', block: 'center'}});
                    clearInterval(id);
                    return;
                }}
                // Fallback: try scrolling main or block-container
                var main = window.parent.document.querySelector('main');
                if (main) {{
                    main.scrollTo({{ top: main.scrollHeight/1.05, behavior: 'smooth' }});
                }}
                var container = window.parent.document.querySelector('.block-container, .stApp, .main');
                if (container) {{
                    container.scrollTo({{ top: container.scrollHeight/1.05, behavior: 'smooth' }});
                }}
            }} catch (e) {{
                // ignore & retry
            }}
            if (attempts >= maxAttempts) clearInterval(id);
        }}, interval);
    }})();
    </script>
    """
    components.html(js, height=0)

# -------------------------
# Response Functions (Updated for multilingual support)
# -------------------------
def get_weather(location, api_key=open_weather_api):
    """
    Get comprehensive weather data with agricultural insights
    """
    current_lang = st.session_state.selected_language
    
    try:
        # Translate location to English for API call
        english_location = translate_text(location, target_lang='en', source_lang='auto')
        
        current_url = f"http://api.openweathermap.org/data/2.5/weather?q={english_location}&appid={api_key}&units=metric"
        forecast_url = f"http://api.openweathermap.org/data/2.5/forecast?q={english_location}&appid={api_key}&units=metric"

        current_response = requests.get(current_url, timeout=10)
        forecast_response = requests.get(forecast_url, timeout=10)

        if current_response.status_code != 200:
            error_msg = f"❌ Unable to fetch weather data for {location}. Please check the location name."
            return translate_text(error_msg, target_lang=current_lang, source_lang='en')

        current_data = current_response.json()
        forecast_data = forecast_response.json() if forecast_response.status_code == 200 else None

        temp = current_data['main']['temp']
        humidity = current_data['main']['humidity']
        pressure = current_data['main']['pressure']
        wind_speed = current_data['wind']['speed']
        description = current_data['weather'][0]['description'].title()
        feels_like = current_data['main']['feels_like']

        sunrise = datetime.fromtimestamp(current_data['sys']['sunrise'])
        sunset = datetime.fromtimestamp(current_data['sys']['sunset'])
        daylight_hours = (sunset - sunrise).seconds / 3600

        rain_1h = current_data.get('rain', {}).get('1h', 0)
        rain_3h = current_data.get('rain', {}).get('3h', 0)

        weather_info = f"""
🌤️ **{get_ui_text('weather_report', current_lang)} {location}**

**{get_ui_text('current_conditions', current_lang)}**
• {get_ui_text('temperature', current_lang)}: {temp}°C (feels like {feels_like}°C)
• {get_ui_text('condition', current_lang)}: {description}
• {get_ui_text('humidity', current_lang)}: {humidity}%
• {get_ui_text('wind_speed', current_lang)}: {wind_speed} m/s
• {get_ui_text('atmospheric_pressure', current_lang)}: {pressure} hPa
• {get_ui_text('daylight_hours', current_lang)}: {daylight_hours:.1f} hours

"""
        if rain_1h > 0 or rain_3h > 0:
            if rain_1h > 0 or rain_3h > 0:
                rainfall_text = f"🌧️ **{get_ui_text('rainfall', current_lang)}:**\n"
            if rain_1h > 0:
                rainfall_text += f"• Last hour: {rain_1h} mm\n"
            if rain_3h > 0:
                rainfall_text += f"• Last 3 hours: {rain_3h} mm\n"
            weather_info += rainfall_text + "\n"

        alerts = generate_agricultural_alerts(temp, humidity, wind_speed, rain_1h, description, current_lang)
        if alerts:
            weather_info += f"🚨 **{get_ui_text('agricultural_alerts', current_lang)}:**\n{alerts}\n"

        if forecast_data:
            insights = generate_predictive_insights(forecast_data, current_lang)
            weather_info += f"\n📊 **3-Day Predictive Insights:**\n{insights}\n"

        recommendations = generate_farming_recommendations(temp, humidity, wind_speed, rain_1h, description, current_lang)
        weather_info += f"\n🌾 **{get_ui_text('farming_recommendations', current_lang)}:**\n{recommendations}"

        # Translate the entire response to the selected language if not English
        if current_lang != 'en':
            weather_info = translate_text(weather_info, target_lang=current_lang, source_lang='en')

        return weather_info

    except requests.exceptions.Timeout:
        error_msg = f"⏱️ Request timed out. Please try again."
        return translate_text(error_msg, target_lang=current_lang, source_lang='en')
    except requests.exceptions.RequestException:
        error_msg = f"🌐 Network error occurred. Please check your internet connection."
        return translate_text(error_msg, target_lang=current_lang, source_lang='en')
    except KeyError:
        error_msg = f"📊 Error processing weather data. Please try again."
        return translate_text(error_msg, target_lang=current_lang, source_lang='en')
    except Exception as e:
        error_msg = f"❌ An unexpected error occurred: {str(e)}"
        return translate_text(error_msg, target_lang=current_lang, source_lang='en')

def generate_agricultural_alerts(temp, humidity, wind_speed, rainfall, description, lang='en'):
    """Generate weather-based agricultural alerts"""
    alerts = []

    # Temperature alerts
    if temp > 35:
        alerts.append("• 🔥 **Heat Stress Alert**: High temperatures may stress crops. Increase irrigation.")
    elif temp < 5:
        alerts.append("• 🧊 **Frost Warning**: Risk of frost damage to sensitive crops.")

    # Humidity alerts
    if humidity > 85:
        alerts.append("• 💧 **High Humidity Alert**: Risk of fungal diseases. Improve ventilation.")
    elif humidity < 30:
        alerts.append("• 🏜️ **Low Humidity Alert**: Crops may need additional water.")

    # Wind alerts
    if wind_speed > 10:
        alerts.append("• 💨 **Strong Wind Alert**: Risk of crop damage and increased evaporation.")

    # Weather condition alerts
    if "storm" in description.lower() or "thunder" in description.lower():
        alerts.append("• ⛈️ **Storm Alert**: Protect crops and secure equipment.")

    if rainfall > 10:
        alerts.append("• 🌊 **Heavy Rain Alert**: Risk of waterlogging and soil erosion.")

    if not alerts:
        alerts.append("• ✅ No immediate weather concerns for farming.")

    alert_text = "\n".join(alerts)
    
    # Translate if not English
    if lang != 'en':
        alert_text = translate_text(alert_text, target_lang=lang, source_lang='en')
    
    return alert_text

def generate_predictive_insights(forecast_data, lang='en'):
    """Generate predictive insights from 5-day forecast"""
    insights = []

    # Analyze next 3 days (24 forecasts ~ 3 days)
    daily_forecasts = {}

    for forecast in forecast_data['list'][:24]:
        date = datetime.fromtimestamp(forecast['dt']).date()

        if date not in daily_forecasts:
            daily_forecasts[date] = {
                'temps': [],
                'humidity': [],
                'rain': 0,
                'conditions': []
            }

        daily_forecasts[date]['temps'].append(forecast['main']['temp'])
        daily_forecasts[date]['humidity'].append(forecast['main']['humidity'])
        daily_forecasts[date]['rain'] += forecast.get('rain', {}).get('3h', 0)
        daily_forecasts[date]['conditions'].append(forecast['weather'][0]['main'])

    for date, data in list(daily_forecasts.items())[:3]:
        avg_temp = sum(data['temps']) / len(data['temps'])
        avg_humidity = sum(data['humidity']) / len(data['humidity'])
        total_rain = data['rain']

        day_name = date.strftime("%A")
        insights.append(f"• **{day_name}**: {avg_temp:.1f}°C, {avg_humidity:.0f}% humidity, {total_rain:.1f}mm rain")

    if len(daily_forecasts) >= 2:
        dates = list(daily_forecasts.keys())
        temp_trend = "increasing" if daily_forecasts[dates[1]]['temps'][0] > daily_forecasts[dates[0]]['temps'][0] else "decreasing"
        insights.append(f"• **Trend**: Temperature is {temp_trend} over the next few days")

    insight_text = "\n".join(insights)
    
    # Translate if not English
    if lang != 'en':
        insight_text = translate_text(insight_text, target_lang=lang, source_lang='en')
    
    return insight_text

def generate_farming_recommendations(temp, humidity, wind_speed, rainfall, description, lang='en'):
    """Generate specific farming recommendations based on weather"""
    recommendations = []

    # Irrigation recommendations
    if temp > 30 and humidity < 50:
        recommendations.append("• 💧 Increase irrigation frequency due to high evaporation rates")
    elif rainfall > 5:
        recommendations.append("• 💧 Reduce or skip irrigation; soil moisture should be adequate")

    # Pest and disease management
    if humidity > 80 and temp > 20:
        recommendations.append("• 🐛 Monitor for fungal diseases; consider preventive fungicide application")

    if wind_speed < 2 and humidity > 70:
        recommendations.append("• 🌬️ Poor air circulation may promote diseases; consider mechanical ventilation")

    # Harvesting recommendations
    if "clear" in description.lower() and wind_speed < 5:
        recommendations.append("• 🌾 Excellent conditions for harvesting and field operations")
    elif "rain" in description.lower():
        recommendations.append("• ⛈️ Avoid heavy machinery operations; soil may be too wet")

    # Planting recommendations
    if 15 <= temp <= 25 and humidity > 40:
        recommendations.append("• 🌱 Good conditions for seed germination and planting")

    if not recommendations:
        recommendations.append("• ✅ Continue with regular farming operations")

    rec_text = "\n".join(recommendations)
    
    # Translate if not English
    if lang != 'en':
        rec_text = translate_text(rec_text, target_lang=lang, source_lang='en')
    
    return rec_text

def get_weather_for_CR(state_name):
    url = f"http://api.openweathermap.org/data/2.5/weather?q={state_name},IN&appid={open_weather_api}&units=metric"
    response = requests.get(url).json()

    # Default values
    temperature = 0.0
    humidity = 0.0
    rainfall = 0.0

    # Only extract if API response has data
    if "main" in response:
        temperature = response["main"].get("temp", 0.0)
        humidity = response["main"].get("humidity", 0.0)

    if "rain" in response:
        rainfall = response["rain"].get("1h", 0.0)  # rainfall in mm

    return temperature, humidity, rainfall

# Main crop recommendation function
def get_crop_recommendation(selected_state):
    """
    Generate a detailed crop recommendation report for the selected state.
    """
    current_lang = st.session_state.selected_language
    
    try:
        # --- Fetch soil data from CSV ---
        state_row = soil_df[soil_df['State'] == selected_state].iloc[0]
        N = state_row['N']
        P = state_row['P']
        K = state_row['K']
        ph = state_row['pH']

        # --- Fetch weather data from API ---
        temperature, humidity, rainfall = get_weather_for_CR(selected_state)

        # --- Predict crop using ML model ---
        recommended_crop = predict_crop(N, P, K, temperature, humidity, ph, rainfall)

        # --- Build detailed formatted report ---
        report = f"""
🌱 **{get_ui_text('crop_recommendation', current_lang)} {selected_state}**

📊 **{get_ui_text('soil_profile', current_lang)}**
• Nitrogen (N): {N}
• Phosphorus (P): {P}
• Potassium (K): {K}
• Soil pH: {ph}

🌤️ **{get_ui_text('weather_conditions', current_lang)}**
• {get_ui_text('temperature', current_lang)}: {temperature} °C
• {get_ui_text('humidity', current_lang)}: {humidity} %
• Rainfall (last 1h): {rainfall} mm

🎯 **{get_ui_text('recommended_crop', current_lang)}**  

**{recommended_crop}**

---
💡 *This recommendation is generated by analyzing both the soil nutrient profile and the latest weather conditions in {selected_state}.* 
"""
        
        # Translate if not English
        if current_lang != 'en':
            # Translate the entire report except for the crop name and numerical values
            report = translate_text(report, target_lang=current_lang, source_lang='en')
        
        return report.strip()

    except IndexError:
        error_msg = f"❌ No soil data found for {selected_state}. Please check your CSV file."
        return translate_text(error_msg, target_lang=current_lang, source_lang='en')
    except Exception as e:
        error_msg = f"❌ An error occurred while generating recommendation: {str(e)}"
        return translate_text(error_msg, target_lang=current_lang, source_lang='en')

def predict_crop(N, P, K, temperature, humidity, ph, rainfall):
    input_df = pd.DataFrame([[N, P, K, temperature, humidity, ph, rainfall]],
                            columns=['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall'])
    input_scaled = scaler.transform(input_df)
    prediction_encoded = model_gbc.predict(input_scaled)
    prediction = encoder.inverse_transform(prediction_encoded)
    return prediction[0]

def get_fertilizer_params_by_state(state_name):
    """
    Fetch temperature, humidity, moisture from OpenWeather API and N, P, K from soil CSV by state.
    Returns: temperature, humidity, moisture, nitrogen, phosphorus, potassium
    """
    current_lang = st.session_state.selected_language
    
    try:
        # --- Fetch weather data from API ---
        url = f"http://api.openweathermap.org/data/2.5/weather?q={state_name},IN&appid={open_weather_api}&units=metric"
        response = requests.get(url, timeout=10)
        
        if response.status_code != 200:
            error_msg = f"❌ Unable to fetch weather data for {state_name}"
            return None, None, None, None, None, None, translate_text(error_msg, target_lang=current_lang, source_lang='en')
        
        weather_data = response.json()
        
        # Extract weather parameters
        temperature = weather_data["main"].get("temp", 0.0)
        humidity = weather_data["main"].get("humidity", 0.0)
        
        # Calculate moisture from humidity and other factors (you can adjust this formula)
        # Using humidity as base and adjusting based on temperature
        moisture = humidity * 0.8 if temperature > 25 else humidity * 0.9
        moisture = min(moisture, 100)  # Cap at 100%
        
        # --- Fetch soil data from CSV ---
        state_row = soil_df[soil_df['State'] == state_name]
        if state_row.empty:
            error_msg = f"❌ No soil data found for {state_name}"
            return temperature, humidity, moisture, None, None, None, translate_text(error_msg, target_lang=current_lang, source_lang='en')
        
        state_row = state_row.iloc[0]
        nitrogen = state_row['N']
        phosphorus = state_row['P']
        potassium = state_row['K']
        
        return temperature, humidity, moisture, nitrogen, phosphorus, potassium, None
        
    except requests.exceptions.Timeout:
        error_msg = "⏱️ Request timed out. Please try again."
        return None, None, None, None, None, None, translate_text(error_msg, target_lang=current_lang, source_lang='en')
    except requests.exceptions.RequestException:
        error_msg = "🌐 Network error occurred. Please check your internet connection."
        return None, None, None, None, None, None, translate_text(error_msg, target_lang=current_lang, source_lang='en')
    except KeyError as e:
        error_msg = f"📊 Error processing data: missing key {str(e)}"
        return None, None, None, None, None, None, translate_text(error_msg, target_lang=current_lang, source_lang='en')
    except Exception as e:
        error_msg = f"❌ An unexpected error occurred: {str(e)}"
        return None, None, None, None, None, None, translate_text(error_msg, target_lang=current_lang, source_lang='en')

def get_fertilizer_recommendation(temperature, humidity, moisture, soil_type, crop_type, nitrogen, phosphorus, potassium):
    """
    Generate fertilizer recommendation using multiple ML models and return detailed analysis.
    """
    current_lang = st.session_state.selected_language
    
    try:
        # Map string values to numeric codes using the dictionaries
        crop_type_encoded = None
        soil_type_encoded = None
        
        # Find the key for crop type
        for key, value in croptype_dict.items():
            if value == crop_type:
                crop_type_encoded = key
                break
                
        # Find the key for soil type  
        for key, value in soiltype_dict.items():
            if value == soil_type:
                soil_type_encoded = key
                break
        
        if crop_type_encoded is None or soil_type_encoded is None:
            error_msg = "❌ Invalid crop type or soil type selected."
            return translate_text(error_msg, target_lang=current_lang, source_lang='en')

        # Prepare input data as numpy array (same as training: X.values)
        # Order: Temperature, Humidity, Moisture, Soil Type, Crop Type, Nitrogen, Potassium, Phosphorous
        input_data = np.array([[temperature, humidity, moisture, soil_type_encoded, crop_type_encoded, nitrogen, potassium, phosphorus]])

        # Get predictions from all three models
        svm_pred = svm_pipeline.predict(input_data)[0]
        rf_pred = rf_pipeline.predict(input_data)[0]
        xgb_pred = xgb_pipeline.predict(input_data)[0]

        # Map predictions to fertilizer names
        svm_fertilizer = fertname_dict[svm_pred]
        rf_fertilizer = fertname_dict[rf_pred]
        xgb_fertilizer = fertname_dict[xgb_pred]

        # Generate ensemble recommendation (most common prediction)
        predictions = [svm_pred, rf_pred, xgb_pred]
        ensemble_pred = max(set(predictions), key=predictions.count)
        ensemble_fertilizer = fertname_dict[ensemble_pred]

        # Build detailed report
        report = f"""
{get_ui_text('fertilizer_rec_report', current_lang)}

📊 **{get_ui_text('environmental_conditions', current_lang)}**
- {get_ui_text('temperature', current_lang)}: {temperature:.1f}°C
- {get_ui_text('humidity', current_lang)}: {humidity:.1f}%
- Soil Moisture: {moisture:.1f}%

🧪 **{get_ui_text('soil_nutrient_profile', current_lang)}**
- Nitrogen (N): {nitrogen}
- Phosphorus (P): {phosphorus}
- Potassium (K): {potassium}

🌾 **{get_ui_text('crop_soil_info', current_lang)}**
- {get_ui_text('crop_type', current_lang)} {crop_type}
- {get_ui_text('soil_type', current_lang)} {soil_type}

🎯 **{get_ui_text('fertilizer_recommendation', current_lang)}**

**{ensemble_fertilizer}**

📋 **{get_ui_text('application_guidelines', current_lang)}**
- Apply fertilizer during early growth stages for maximum efficiency
- Follow manufacturer's dosage instructions
- Consider current soil moisture ({moisture:.1f}%) before application
- Monitor crop response and adjust as needed

⚠️ **{get_ui_text('important_notes', current_lang)}**
- Conduct soil testing regularly for precise nutrient management
- Consider organic alternatives when possible
- Avoid over-fertilization to prevent environmental damage

---
💡 *This recommendation is based on real-time weather data and soil analysis for optimal fertilizer selection.*
"""
        
        # Translate if not English
        if current_lang != 'en':
            report = translate_text(report, target_lang=current_lang, source_lang='en')
        
        return report.strip()

    except Exception as e:
        error_msg = f"❌ An error occurred while generating fertilizer recommendation: {str(e)}"
        return translate_text(error_msg, target_lang=current_lang, source_lang='en')

def get_market_price_prediction(date_str, district, commodity):
    current_lang = st.session_state.selected_language
    
    try:
        # Translate district and commodity to English for processing
        english_district = translate_text(district, target_lang='en', source_lang='auto')
        english_commodity = translate_text(commodity, target_lang='en', source_lang='auto')
        
        input_date = datetime.strptime(date_str, "%d-%m-%Y").date()

        subset = price_df[(price_df['District'] == english_district) & (price_df['Commodity'] == english_commodity)]
        if subset.empty:
            error_msg = f"⚠️ No data available for {commodity} in {district}."
            return translate_text(error_msg, target_lang=current_lang, source_lang='en')

        market = subset['Market'].mode()[0]
        state = subset['State'].iloc[0]

        input_features = pd.DataFrame({
            'day': [input_date.day],
            'month': [input_date.month],
            'year': [input_date.year],
            'State_enc': [le_state.transform([state])[0]],
            'District_enc': [le_district.transform([english_district])[0]],
            'Market_enc': [le_market.transform([market])[0]],
            'Commodity_enc': [le_commodity.transform([english_commodity])[0]]
        })

        predicted_price = model_cr_pr.predict(input_features)[0]

        result = (
            f"📊 {get_ui_text('market_price_prediction', current_lang)} \n\n"
            f"- 📍 State: {state}\n"
            f"- 🏢 District: {district}\n"
            f"- 🛒 Market: {market}\n"
            f"- 🌱 Commodity: {commodity}\n"
            f"- 📅 Date: {input_date.strftime('%d-%m-%Y')}\n\n"
            f"💰 **{get_ui_text('predicted_modal_price', current_lang)}**: ₹{predicted_price:.2f}"
        )
        
        # Translate if not English
        if current_lang != 'en':
            result = translate_text(result, target_lang=current_lang, source_lang='en')
        
        return result

    except Exception as e:
        error_msg = f"❌ Error: {str(e)}"
        return translate_text(error_msg, target_lang=current_lang, source_lang='en')

def get_response(intent):
    current_lang = st.session_state.selected_language
    
    if intent == "greetings":
        response = "Hello! How can I help you today?"
    elif intent == "fallback":
        response = get_ui_text('fallback_response', current_lang)
        return response
    else:
        response = get_ui_text('fallback_response', current_lang)
        return response
    
    # Translate if not English
    if current_lang != 'en':
        response = translate_text(response, target_lang=current_lang, source_lang='en')
    
    return response

# -------------------------
# Initialize Session
# -------------------------
def initialize_session_state():
    if "history" not in st.session_state:
        current_lang = st.session_state.get('selected_language', 'en')
        greeting = get_ui_text('greeting_message', current_lang)
        st.session_state.history = [Message("ai", greeting)]
    if "awaiting_weather" not in st.session_state:
        st.session_state.awaiting_weather = False
    if "awaiting_crop" not in st.session_state:
        st.session_state.awaiting_crop = False
    if "awaiting_fertilizer_pesticide_advice" not in st.session_state:
        st.session_state.awaiting_fertilizer_pesticide_advice = False

# -------------------------
# Streamlit App
# -------------------------
st.set_page_config(page_title="KrishiMitra", page_icon="static/logo.png", layout="wide", initial_sidebar_state="collapsed")
load_css()
initialize_language_session()
initialize_session_state()

#st.title("KrishiMitra")
chat_placeholder = st.container()

# Display chat history
with chat_placeholder:
    for chat in st.session_state.history:
        div = f"""
<div class="chat-row {'row-reverse' if chat.origin == 'human' else ''}">
    <img class="chat-icon" src="app/static/{'farmer.png' if chat.origin == 'human' else 'bot.png'}" width=32 height=32>
    <div class="chat-bubble {'human-bubble' if chat.origin == 'human' else 'ai-bubble'}">
        &#8203;{chat.message}
    </div>
</div>
        """
        st.markdown(div, unsafe_allow_html=True)

    st.markdown("")  # spacer

    # Auto-scroll only when AI sends the last message
    if st.session_state.history and st.session_state.history[-1].origin == "ai":
        scroll_to_bottom()

# -------------------------
# Dynamic Inputs based on Bot Request (Updated for multilingual support)
# -------------------------
current_lang = st.session_state.selected_language

if st.session_state.awaiting_weather:
    # place an anchor just above the input so we can scroll to it
    with chat_placeholder:
        st.markdown('<div id="input-anchor-weather"></div>', unsafe_allow_html=True)
        # prompt + input
        st.markdown(f"**{get_ui_text('location_prompt', current_lang)}**")
        cols = st.columns((6, 1))

        # Ensure the anchor is visible
        scroll_to_anchor("input-anchor-weather")

        # Text input for user location
        location = cols[0].text_input(
            get_ui_text('location', current_lang),
            value="",
            label_visibility="collapsed",
            key="weather_prompt"
        )

        # Submit button
        if cols[1].button(get_ui_text('submit', current_lang)):
            if location.strip():
                # Detect language of user input
                detected_lang = detect_language(location)
                st.session_state.user_input_language = detected_lang
                
                bot_reply = get_weather(location.strip())
                st.session_state.history.append(Message("human", location.strip()))
                st.session_state.history.append(Message("ai", bot_reply))
                st.session_state.awaiting_weather = False
                st.rerun()

elif st.session_state.awaiting_crop:
    # place an anchor above the state selectbox
    with chat_placeholder:
        st.markdown('<div id="input-anchor-crop"></div>', unsafe_allow_html=True)
        states = soil_df['State'].tolist()
        st.markdown(f"**{get_ui_text('location_prompt', current_lang)}**")
        cols = st.columns((6, 1))

        # Ensure the anchor is visible
        scroll_to_anchor("input-anchor-crop")

        # Selectbox for state
        selected_state = cols[0].selectbox(get_ui_text('select_state', current_lang), states, key="crop_state_select")

        # Submit button
        if cols[1].button(get_ui_text('submit', current_lang)):
            # show the user's selection in chat history for clarity
            st.session_state.history.append(Message("human", selected_state))
            bot_reply = get_crop_recommendation(selected_state)
            st.session_state.history.append(Message("ai", bot_reply))
            st.session_state.awaiting_crop = False
            st.rerun()

elif st.session_state.awaiting_fertilizer_pesticide_advice:
    # place an anchor above the fertilizer inputs
    with chat_placeholder:
        st.markdown('<div id="input-anchor-fertilizer"></div>', unsafe_allow_html=True)
        st.markdown(f"**{get_ui_text('fertilizer_form_title', current_lang)}**")
        
        # Ensure the anchor is visible
        scroll_to_anchor("input-anchor-fertilizer")

        with st.form("fertilizer_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                # State selection (auto-fetch weather and soil data)
                states = soil_df['State'].tolist()
                selected_state = st.selectbox(get_ui_text('select_state', current_lang), states)
                
                # Get unique crop types and soil types from the dictionaries
                crop_types = list(croptype_dict.values())
                soil_types = list(soiltype_dict.values())
                
                crop_type = st.selectbox(get_ui_text('crop_type', current_lang), crop_types)
                soil_type = st.selectbox(get_ui_text('soil_type', current_lang), soil_types)
            
            with col2:
                auto_fetch_text = "**Auto-fetched parameters:**\nTemperature, Humidity, Moisture will be fetched from weather API\nN, P, K will be fetched from soil database"
                if current_lang != 'en':
                    auto_fetch_text = translate_text(auto_fetch_text, target_lang=current_lang, source_lang='en')
                st.markdown(auto_fetch_text)
                st.info(auto_fetch_text)
            
            submitted = st.form_submit_button(get_ui_text('get_fertilizer_rec', current_lang))
            
            if submitted:
                # Fetch parameters automatically
                temperature, humidity, moisture, nitrogen, phosphorus, potassium, error = get_fertilizer_params_by_state(selected_state)
                
                if error:
                    st.session_state.history.append(Message("ai", error))
                else:
                    # Show user inputs in chat history
                    user_input = f"State: {selected_state}, Crop: {crop_type}, Soil: {soil_type}"
                    st.session_state.history.append(Message("human", user_input))
                    
                    bot_reply = get_fertilizer_recommendation(temperature, humidity, moisture, soil_type, crop_type, nitrogen, phosphorus, potassium)
                    st.session_state.history.append(Message("ai", bot_reply))
                
                st.session_state.awaiting_fertilizer_pesticide_advice = False
                st.rerun()

elif st.session_state.get("awaiting_market_price_info", False):
    with chat_placeholder:
        st.markdown('<div id="input-anchor-market"></div>', unsafe_allow_html=True)
        st.markdown(f"**{get_ui_text('market_price_form', current_lang)}**")
        
        scroll_to_anchor("input-anchor-market")
        
        with st.form("market_price_form"):
            date_str = st.text_input(get_ui_text('enter_date', current_lang))
            district = st.text_input(get_ui_text('enter_district', current_lang))
            commodity = st.text_input(get_ui_text('enter_commodity', current_lang))
            submitted = st.form_submit_button(get_ui_text('get_price_prediction', current_lang))

            if submitted:
                user_input = f"Date: {date_str}, District: {district}, Commodity: {commodity}"
                st.session_state.history.append(Message("human", user_input))
                
                bot_reply = get_market_price_prediction(date_str, district, commodity)
                st.session_state.history.append(Message("ai", bot_reply))
                st.session_state.awaiting_market_price_info = False
                st.rerun()

# -------------------------
# Crop Disease Detection Handler
# -------------------------
elif st.session_state.get("awaiting_disease_detection", False):
    with chat_placeholder:
        st.markdown('<div id="input-anchor-disease"></div>', unsafe_allow_html=True)
        st.markdown(f"**{get_ui_text('disease_detection_prompt', current_lang)}**")
        cols = st.columns((6, 1))

        scroll_to_anchor("input-anchor-disease")

        uploaded_file = cols[0].file_uploader(get_ui_text('choose_image', current_lang), type=["jpg", "jpeg", "png"], key="disease_image")
        if cols[1].button(get_ui_text('submit', current_lang)) and uploaded_file:
            try:
                disease_name, confidence = crop_model_prediction(uploaded_file)

                bot_reply = f"""
🌱 **{get_ui_text('crop_disease_result', current_lang)}**

**{get_ui_text('detected_disease', current_lang)}** 🩺 **{disease_name.replace('_', ' ')}**  
**{get_ui_text('confidence_level', current_lang)}** 💯 **{confidence:.2f}%**

⚠️ **{get_ui_text('recommendation', current_lang)}** Inspect the affected crop closely and take necessary measures such as consulting an agronomist, applying appropriate treatment, or isolating affected plants to prevent spread.

📌 **Note:** Regular monitoring and preventive care help minimize disease impact and improve crop health.
"""
                
                # Translate if not English
                if current_lang != 'en':
                    bot_reply = translate_text(bot_reply, target_lang=current_lang, source_lang='en')

                st.session_state.history.append(Message("ai", bot_reply))

            except Exception as e:
                error_msg = f"❌ Error during disease prediction: {str(e)}"
                if current_lang != 'en':
                    error_msg = translate_text(error_msg, target_lang=current_lang, source_lang='en')
                st.session_state.history.append(Message("ai", error_msg))

            st.session_state.awaiting_disease_detection = False
            st.rerun()

elif st.session_state.get("awaiting_soil_health", False):
    with chat_placeholder:
        st.markdown('<div id="input-anchor-soil"></div>', unsafe_allow_html=True)
        st.markdown(f"**{get_ui_text('soil_health_form', current_lang)}**")
        
        scroll_to_anchor("input-anchor-soil")
        
        with st.form("soil_health_form"):
            soil_ph = st.number_input(get_ui_text('soil_ph', current_lang), min_value=3.0, max_value=10.0, value=7.0, step=0.1)
            organic_matter = st.number_input(get_ui_text('organic_matter', current_lang), min_value=0.0, max_value=20.0, value=3.0, step=0.1)
            soil_moisture = st.number_input(get_ui_text('soil_moisture', current_lang), min_value=0, max_value=100, value=50)
            submitted = st.form_submit_button(get_ui_text('get_soil_analysis', current_lang))
            
            if submitted:
                user_input = f"pH: {soil_ph}, Organic Matter: {organic_matter}%, Moisture: {soil_moisture}%"
                st.session_state.history.append(Message("human", user_input))
                
                # Generate soil health analysis
                bot_reply = f"""
🌱 **{get_ui_text('soil_health_report', current_lang)}**

📊 **{get_ui_text('soil_parameters', current_lang)}**
• pH Level: {soil_ph}
• {get_ui_text('organic_matter', current_lang)}: {organic_matter}%
• {get_ui_text('soil_moisture', current_lang)}: {soil_moisture}%

🔍 **{get_ui_text('analysis', current_lang)}**
• pH Status: {'Acidic' if soil_ph < 6.5 else 'Neutral' if soil_ph <= 7.5 else 'Alkaline'}
• {get_ui_text('organic_matter', current_lang)}: {'Low' if organic_matter < 2 else 'Good' if organic_matter <= 5 else 'Excellent'}
• Moisture Level: {'Dry' if soil_moisture < 30 else 'Optimal' if soil_moisture <= 70 else 'Wet'}

💡 **{get_ui_text('recommendations', current_lang)}**
{'• Consider adding lime to reduce acidity' if soil_ph < 6.0 else '• Consider adding sulfur to reduce alkalinity' if soil_ph > 8.0 else '• pH level is in good range'}
{'• Add compost or organic matter to improve soil structure' if organic_matter < 3 else '• Maintain current organic matter levels'}
{'• Increase irrigation frequency' if soil_moisture < 30 else '• Reduce irrigation to prevent waterlogging' if soil_moisture > 80 else '• Maintain current irrigation schedule'}

---
💡 *Regular soil testing helps maintain optimal growing conditions for crops.*
"""
                
                # Translate if not English
                if current_lang != 'en':
                    bot_reply = translate_text(bot_reply, target_lang=current_lang, source_lang='en')
                
                st.session_state.history.append(Message("ai", bot_reply))
                st.session_state.awaiting_soil_health = False
                st.rerun()

# -------------------------
# Normal Chat Input (Updated for multilingual support)
# -------------------------
else:
    # Text chat input at the bottom
    if prompt := st.chat_input(get_ui_text('type_message', current_lang)):
        # Detect language of user input
        detected_lang = detect_language(prompt)
        st.session_state.user_input_language = detected_lang
        
        st.session_state.history.append(Message("human", prompt))
        intent = predict_intent(prompt)
        
        if intent == "weather_query":
            st.session_state.awaiting_weather = True
            response = get_ui_text('weather_location_request', current_lang)
            st.session_state.history.append(Message("ai", response))
        elif intent == "crop_recommendation":
            st.session_state.awaiting_crop = True
            response = get_ui_text('crop_location_request', current_lang)
            st.session_state.history.append(Message("ai", response))
        elif intent == "disease_detection":
            st.session_state.awaiting_disease_detection = True
            response = get_ui_text('disease_image_request', current_lang)
            st.session_state.history.append(Message("ai", response))
        elif intent == "fertilizer_pesticide_advice":
            st.session_state.awaiting_fertilizer_pesticide_advice = True
            response = get_ui_text('fertilizer_info_request', current_lang)
            st.session_state.history.append(Message("ai", response))
        elif intent == "soil_health":
            st.session_state.awaiting_soil_health = True
            response = get_ui_text('soil_health_request', current_lang)
            st.session_state.history.append(Message("ai", response))
        elif intent == "market_price_info":
            st.session_state.awaiting_market_price_info = True
            response = get_ui_text('market_info_request', current_lang)
            st.session_state.history.append(Message("ai", response))
        else:
            response = get_response(intent)
            st.session_state.history.append(Message("ai", response))
        

        st.rerun()




