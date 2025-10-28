import streamlit as st
import joblib

# Load model (pipeline includes vectorizer)
model = joblib.load("models/Emotion_model.pkl")

# Page config
st.set_page_config(page_title="Indhra AI", layout="wide")

# ✅ Modern & fixed CSS styling
st.markdown("""
<style>
/* Full app background */
.stApp {
    background: linear-gradient(to top, #0c3483 0%, #6b8cce 50%, #a2b6df 100%);
    color: #1b1b1b;
    font-family: 'Poppins', sans-serif;
}

/* Title & subtitle styling */
.main-title {
    font-size: 48px;
    font-weight: 700;
    text-align: center;
    color: #ffffff;
    text-shadow: 0 0 12px rgba(0,0,0,0.3);
    margin-bottom: 10px;
}
.subtitle {
    font-size: 18px;
    text-align: center;
    color: #e0e6f8;
    margin-bottom: 40px;
}
/* Style for "Enter your sentence:" label */
label[data-testid="stMarkdownContainer"] p {
    font-size: 24px !important;   /* increase label font size */
    color: #ffffff !important;    /* make text white */
    font-weight: 600 !important;
}

/* Label for text area */
label[data-testid="stMarkdownContainer"] p {
    font-size: 22px !important;
    color: #ffffff !important;
    font-weight: 600 !important;
}

/* Button Style */
.stButton > button {
    background-image: linear-gradient(135deg, #5adaff 0%, #5468ff 100%);
    color: white !important;
    border: none;
    border-radius: 10px;
    padding: 14px 30px;
    font-size: 20px;
    font-weight: 600;
    transition: all 0.3s ease;
    cursor: pointer;
    box-shadow: 0 4px 10px rgba(0,0,0,0.3);
}
.stButton > button:hover {
    transform: scale(1.05);
    background-image: linear-gradient(135deg, #6fb1fc 0%, #4364f7 100%);
    color: white !important; /* Keeps text white even when hovered */
}

/* Textarea style */
textarea {
    font-size: 18px !important;
    border-radius: 8px !important;
    background-color: rgba(255,255,255,0.2) !important;
    color: #1b1b1b !important;
    font-weight: 500 !important;
}

/* Result box */
.result-box {
    background: rgba(255, 255, 255, 0.15);
    border-radius: 12px;
    padding: 25px;
    margin-top: 25px;
    text-align: center;
    border: 1px solid rgba(255,255,255,0.3);
    box-shadow: 0px 4px 12px rgba(0,0,0,0.3);
    max-width: 600px;
    margin-left: auto;
    margin-right: auto;
    backdrop-filter: blur(10px);
}
.emotion-text {
    font-size: 36px;
    font-weight: 700;
    color: #ffffff;
    letter-spacing: 1px;
}
</style>
""", unsafe_allow_html=True)

# Title & Description
st.markdown("<div class='main-title'>Emotion Detector AI</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Detect human emotions from text message — basic version (1.0) . <br> Predictions may not always be accurate; upgraded version coming soon.</div>", unsafe_allow_html=True)

# User Input
user_input = st.text_area("Enter your sentence:", placeholder="Type something like — I'm really happy today!")

# Detect Emotion Button
if st.button("Detect Emotion"):
    if user_input.strip() == "":
        st.warning("Please enter a sentence to analyze emotion!")
    else:
        prediction = model.predict([user_input])[0]

        # Color mapping for emotion
        color_map = {
            "anger": "#ff6b6b",
            "fear": "#f0a500",
            "joy": "#51cf66"
        }

        st.markdown(f"""
            <div class='result-box'>
                <span class='emotion-text' style='color:{color_map.get(prediction, "#ff4b2b")}'>{prediction.upper()}</span>
            </div>
        """, unsafe_allow_html=True)
