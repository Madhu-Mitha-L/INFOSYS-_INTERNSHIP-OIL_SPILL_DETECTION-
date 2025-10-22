import streamlit as st
import tensorflow as tf
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import io
from tensorflow.keras import backend as K
from datetime import datetime
import sqlite3
import hashlib
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import urllib.parse
from datetime import datetime
import os
import base64

# Page config
st.set_page_config(
    page_title="Oil Spill Detection",
    page_icon="🛢️",
    layout="wide"
)

st.markdown("""
    <div class="header-container">
        <h1 class="main-header">🌊 Oil Spill Detection System</h1>
        <p class="subheader-text">AI-powered detection, monitoring & analysis</p>
    </div>
""", unsafe_allow_html=True)
# Custom CSS
st.markdown("""
    <style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800&display=swap');
    
    /* Global Styles */
    * {
        font-family: 'Poppins', sans-serif;
        margin: 0;
        padding: 0;
    }
    
    /* Main Background with Ocean Theme */
    .stApp {
        background: linear-gradient(135deg, cyan 0%, magenta 50%, yellow 100%) !important; 
        background-attachment: fixed;
    }
    
    /* Add animated ocean background */
    .stApp::before {
        content: "";
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: 
            radial-gradient(ellipse at 50% 50%, rgba(0, 255, 255, 0.05) 0%, transparent 50%),
            radial-gradient(ellipse at 80% 20%, rgba(255, 215, 0, 0.05) 0%, transparent 50%);
        pointer-events: none;
        z-index: 0;
        animation: pulse 15s ease-in-out infinite;
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 0.5; }
        50% { opacity: 0.8; }
    }
    
    /* Main Content Container */
    .block-container {
        background-image: url("https://onlinelibrary.wiley.com/cms/asset/edbe9180-7143-42ab-ba55-33c1223fcc90/js3296495-fig-0005b-m.jpg");
        background-size: cover;
        background-position: center;
        display: flex;
        justify-content: center;
        align-items: center;
        backdrop-filter: blur(20px);
        border-radius: 20px;
        padding: 2rem;
        box-shadow: 0 20px 60px rgba(0, 0, 0, 0.4);
        margin-top: 2rem;
        margin-bottom: 4rem;
        position: relative;
        z-index: 1;
    }
    
    /* Header Styling */
    .main-header {
        font-size: 3.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #FFD700 0%, #FF6B35 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1rem;
        text-shadow: none;
        animation: fadeInDown 1s ease-out;
    }
    
    .subheader-text {
        color: #00d4ff;
        text-align: center;
        font-size: 1rem;
        margin-top: 0.5rem;
        font-weight: 600;
        text-shadow: 0 0 20px rgba(0, 212, 255, 0.3);
    }
    
    @keyframes fadeInDown {
        from {
            opacity: 0;
            transform: translateY(-30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    /* Subheader Styling */
    h2, h3 {
        color: #FFD700;
        font-weight: 700;
    }
    
    /* Header Container */
    .header-container {
        background: linear-gradient(90deg, #1a1a2e 0%, #16213e 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
        border-top: 4px solid #FFD700;
        border-bottom: 4px solid #00d4ff;
        text-align: center;
    }
    
    /* Upload Text */
    .upload-text {
        font-size: 1.3rem;
        color: #00d4ff;
        font-weight: 700;
        text-align: center;
        margin: 1.5rem 0;
        text-shadow: 0 0 20px rgba(0, 212, 255, 0.3);
    }
    
    /* Buttons with Glow Effect, single line */
.stButton>button {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 10px; /* space between icon and text */
    
    background: linear-gradient(135deg, #FFD700 0%, #FF6B35 100%);
    color: white;
    border: none;
    border-radius: 12px;
    padding: 0.5rem 1.5rem; /* reduce height */
    font-size: 20px;        /* smaller font to fit single line */
    font-weight: 700;
    cursor: pointer;
    transition: all 0.3s ease;
    box-shadow: 0 0 30px rgba(255, 215, 0, 0.5), 0 4px 15px rgba(255, 107, 53, 0.4);
    position: relative;
    overflow: hidden;
    text-transform: uppercase;
    letter-spacing: 1px;
    line-height: 1;          /* force single line */
}

    
    .stButton>button:hover {
        transform: translateY(-3px);
        box-shadow: 0 0 50px rgba(0, 212, 255, 0.8), 0 8px 25px rgba(255, 215, 0, 0.6);
        background: linear-gradient(135deg, #FF6B35 0%, #FFD700 100%);
        text-shadow: 0 0 10px rgba(0, 212, 255, 0.5);
    }
    
    .stButton>button:active {
        transform: translateY(-1px);
    }
    
    .stButton>button::before {
        content: "";
        position: absolute;
        top: 50%;
        left: 50%;
        width: 0;
        height: 0;
        border-radius: 50%;
        background: rgba(0, 212, 255, 0.3);
        transform: translate(-50%, -50%);
        transition: width 0.6s, height 0.6s;
    }
    
    .stButton>button:hover::before {
        width: 300px;
        height: 300px;
    }
    
    /* Alert Boxes with Animation */
    .alert-danger {
        background: linear-gradient(135deg, #FF6B35 0%, #E63946 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        font-weight: 700;
        text-align: center;
        box-shadow: 0 0 30px rgba(255, 107, 53, 0.5), 0 8px 20px rgba(230, 57, 70, 0.4);
        animation: slideIn 0.5s ease-out;
        border: 2px solid rgba(0, 212, 255, 0.4);
    }
    
    .alert-warning {
        background: linear-gradient(135deg, #FFD700 0%, #FFA500 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        font-weight: 700;
        text-align: center;
        box-shadow: 0 0 30px rgba(255, 215, 0, 0.5), 0 8px 20px rgba(255, 165, 0, 0.4);
        animation: slideIn 0.5s ease-out;
        border: 2px solid rgba(0, 212, 255, 0.4);
    }
    
    .alert-success {
        background: linear-gradient(135deg, #00C851 0%, #007E33 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        font-weight: 700;
        text-align: center;
        box-shadow: 0 0 30px rgba(0, 212, 255, 0.5), 0 8px 20px rgba(0, 200, 81, 0.4);
        animation: slideIn 0.5s ease-out;
        border: 2px solid rgba(0, 212, 255, 0.4);
    }
    
    @keyframes slideIn {
        from {
            opacity: 0;
            transform: translateX(-50px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }
    
    /* Summary Box */
    .summary-box {
        background: rgba(255, 255, 255, 0.95);
        border: 2px solid #FFD700;
        border-radius: 15px;
        padding: 2rem;
        box-shadow: 0 0 40px rgba(255, 215, 0, 0.3), 0 8px 32px rgba(0, 212, 255, 0.2);
        margin: 1rem 0;
        animation: fadeIn 0.5s ease-out;
    }
    
    /* Metrics Card */
    .metrics-card {
        background: linear-gradient(135deg, #f8f9fa 0%, #ffffff 100%);
        border: 2px solid #FFD700;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 0.5rem 0;
        box-shadow: 0 0 20px rgba(255, 215, 0, 0.2);
        transition: all 0.3s ease;
        font-weight: 600;
    }
    
    .metrics-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 0 40px rgba(0, 212, 255, 0.4);
        border-color: #00d4ff;
    }
    
    /* History Cards with Hover Effect */
    .history-card {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 0 25px rgba(255, 215, 0, 0.2), 0 4px 15px rgba(0,0,0,0.1);
        margin-bottom: 1.5rem;
        border-left: 5px solid #FFD700;
        border: 2px solid rgba(0, 212, 255, 0.3);
        transition: all 0.3s ease;
        animation: fadeIn 0.5s ease-out;
    }
    
    .history-card:hover {
        transform: translateY(-5px) scale(1.02);
        box-shadow: 0 0 50px rgba(0, 212, 255, 0.6), 0 12px 30px rgba(255, 215, 0, 0.3);
        border-left: 5px solid #00d4ff;
    }
    
    @keyframes fadeIn {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
  
    
    @keyframes zoomIn {
        from {
            opacity: 0;
            transform: scale(0.9);
        }
        to {
            opacity: 1;
            transform: scale(1);
        }
    }
    
    /* Input Fields Styling */
    .stTextInput>div>div>input, 
    .stTextArea>div>div>textarea,
    .stSelectbox>div>div>select {
        border-radius: 10px;
        border: 2px solid #FFD700;
        padding: 0.75rem;
        font-size: 1rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stTextInput>div>div>input:focus,
    .stTextArea>div>div>textarea:focus,
    .stSelectbox>div>div>select:focus {
        border-color: #00d4ff;
        box-shadow: 0 0 20px rgba(0, 212, 255, 0.5);
    }
    
    /* Metrics Container */
    [data-testid="metric-container"] {
        background: linear-gradient(135deg, #f8f9fa 0%, #ffffff 100%);
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 0 25px rgba(255, 215, 0, 0.2);
        transition: all 0.3s ease;
        border: 2px solid #FFD700;
    }
    
    [data-testid="metric-container"]:hover {
        transform: translateY(-5px);
        box-shadow: 0 0 40px rgba(0, 212, 255, 0.5);
        border-color: #00d4ff;
    }
    
    [data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: 800;
        color: #FF6B35;
    }
    
    [data-testid="stMetricLabel"] {
        font-size: 35px;
        color: yellow;
        font-weight: 1500;
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(135deg, cyan 50%, magenta 50%, yellow 100%);
        border-right: 3px solid #FFD700;
    }
    
    [data-testid="stSidebar"] .stMarkdown {
        color: black;
            font-weight:bold;
    }
    
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: #FFD700;
        font-weight: 700;
    }
    
    /* File Uploader Styling */
    [data-testid="stFileUploader"] {
        background: linear-gradient(135deg, #a5c5e6 0%, #f1dcdc 100%);
        border: 3px dashed #FFD700;
        border-radius: 15px;
        padding: 2rem;
        transition: all 0.3s ease;
    }
    
    [data-testid="stFileUploader"]:hover {
        border-color: #00d4ff;
        background: linear-gradient(135deg, #a5c5e6 0%, #f1dcdc 100%);
        transform: scale(1.02);
        box-shadow: 0 0 40px rgba(0, 212, 255, 0.4);
    }
    
    /* Download Button Special Styling */
    .stDownloadButton>button {
        background: linear-gradient(135deg, #00C851 0%, #007E33 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 2rem;
        font-weight: 700;
        transition: all 0.3s ease;
        box-shadow: 0 0 30px rgba(0, 200, 81, 0.4);
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .stDownloadButton>button:hover {
        transform: translateY(-3px);
        box-shadow: 0 0 50px rgba(0, 212, 255, 0.8), 0 8px 25px rgba(0, 200, 81, 0.5);
    }
    
    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: transparent;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: linear-gradient(135deg, rgba(255, 215, 0, 0.1), rgba(0, 212, 255, 0.05));
        border-radius: 10px;
        padding: 12px 24px;
        font-weight: 700;
        transition: all 0.3s ease;
        color: #FFD700;
        border-bottom: 2px solid transparent;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: linear-gradient(135deg, rgba(255, 215, 0, 0.2), rgba(0, 212, 255, 0.1));
        transform: translateY(-2px);
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #FFD700 0%, #FF6B35 100%);
        color: white;
        font-weight: 800;
        border-bottom: 3px solid #00d4ff;
    }
    
    /* Image Container */
    .stImage {
        border-radius: 15px;
        overflow: hidden;
        box-shadow: 0 0 30px rgba(255, 215, 0, 0.2), 0 8px 25px rgba(0,0,0,0.15);
        transition: all 0.3s ease;
        border: 2px solid #FFD700;
    }
    
    .stImage:hover {
        transform: scale(1.02);
        box-shadow: 0 0 50px rgba(0, 212, 255, 0.5);
        border-color: #00d4ff;
    }
    
    /* Caption Styling */
    .caption {
        color: #FFD700;
        font-size: 0.9rem;
        text-align: center;
        margin-top: 0.5rem;
        font-weight: 600;
    }
    
    /* Info/Success/Warning Boxes */
    .stAlert {
        border-radius: 12px;
        border-left: 4px solid #FFD700;
        animation: slideIn 0.5s ease-out;
        border: 2px solid #FFD700;
    }
    
    /* Spinner Styling */
    .stSpinner > div {
        border-top-color: #00d4ff !important;
    }
    
    .footer {
    width: 100%;
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    color: white;
    text-align: center;
    padding: 1rem 0;
    border-top: 3px solid #FFD700;
    margin-top: 2rem;
}

/* Remove extra scroll space */
body, .stApp {
    margin: 0;
    padding: 0;
    min-height: 100vh;
    display: flex;
    flex-direction: column;
}

.main-content {
    flex: 1;
}

    
    
    .footer-text {
        margin: 0.5rem 0;
        font-weight: 600;
        color: #00d4ff;
    }
    
    .footer-text-main {
        color: #FFD700;
        font-weight: 700;
        font-size: 1rem;
    }
    
    /* Download Section */
    .download-section {
        background: linear-gradient(135deg, rgba(255, 215, 0, 0.1), rgba(0, 212, 255, 0.05));
        border: 2px dashed #FFD700;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
        transition: all 0.3s ease;
    }
    
    .download-section:hover {
        border-color: #00d4ff;
        box-shadow: 0 0 30px rgba(0, 212, 255, 0.3);
    }
    
    /* Scrollbar Styling */
    ::-webkit-scrollbar {
        width: 10px;
        height: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: #1a1a2e;
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #FFD700 0%, #FF6B35 100%);
        border-radius: 10px;
        box-shadow: 0 0 20px rgba(255, 215, 0, 0.5);
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #00d4ff 0%, #FFD700 100%);
        box-shadow: 0 0 30px rgba(0, 212, 255, 0.6);
    }
    
    /* Navigation Buttons - Same Line */
    .nav-buttons-container {
        display: flex;
        gap: 1rem;
        justify-content: center;
        align-items: center;
        margin-bottom: 1rem;
        flex-wrap: wrap;
    }
    
    /* Responsive Design */
    @media (max-width: 768px) {
        .main-header {
            font-size: 2rem;
        }
        
        .login-box {
            padding: 1.5rem;
            margin: 1rem auto;
        }
        
        .history-card {
            padding: 1rem;
        }
        
        .nav-buttons-container {
            flex-direction: column;
        }
    }
    
    /* Glow Effect */
    .glow {
        animation: glow 2s ease-in-out infinite;
    }
    
    @keyframes glow {
        0%, 100% {
            box-shadow: 0 0 20px rgba(0, 212, 255, 0.5);
        }
        50% {
            box-shadow: 0 0 40px rgba(0, 212, 255, 0.8);
        }
    }
    
    /* Navigation Button Styling */
    .stButton.nav-button>button {
        min-width: 120px;
        font-size: 0.95rem;
    }
            
            /* Slider Track */
.css-1n76uvr.edgvbvh3 {  /* This class is the slider container in Streamlit 1.30+ */
    background: linear-gradient(135deg, #FFD1DC 0%, #FFB6C1 100%) !important; /* pink gradient */
    border-radius: 10px;
}

/* Slider Handle */
.css-15tx938.e1ggruw40 {  /* The draggable knob */
    background-color: #FF69B4 !important;  /* bright pink handle */
}
            
    


            
    </style>
""", unsafe_allow_html=True)



# Initialize session state
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'username' not in st.session_state:
    st.session_state.username = None
if 'user_id' not in st.session_state:
    st.session_state.user_id = None
if 'page' not in st.session_state:
    st.session_state.page = 'login'

# Model Settings
IMG_SIZE = (320, 320)

# ==================== DATABASE FUNCTIONS ====================

def init_db():
    """Initialize SQLite database"""
    conn = sqlite3.connect('oil_spill_detection.db')
    c = conn.cursor()
    
    # Users table
    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  username TEXT UNIQUE NOT NULL,
                  email TEXT NOT NULL,
                  password TEXT NOT NULL,
                  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
    
    # History table - UPDATED with image_data column
    c.execute('''CREATE TABLE IF NOT EXISTS history
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  user_id INTEGER NOT NULL,
                  image_name TEXT NOT NULL,
                  image_data BLOB,
                  oil_spill_area REAL,
                  confidence REAL,
                  severity TEXT,
                  affected_area_km2 REAL,
                  timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                  FOREIGN KEY (user_id) REFERENCES users(id))''')
    
    # Feedback table
    c.execute('''CREATE TABLE IF NOT EXISTS feedback
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  user_id INTEGER NOT NULL,
                  username TEXT NOT NULL,
                  feedback_text TEXT NOT NULL,
                  timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                  FOREIGN KEY (user_id) REFERENCES users(id))''')
    
    
    conn.commit()
    conn.close()

def hash_password(password):
    """Hash password using SHA256"""
    return hashlib.sha256(password.encode()).hexdigest()

def register_user(username, email, password):
    """Register new user"""
    try:
        conn = sqlite3.connect('oil_spill_detection.db')
        c = conn.cursor()
        
        hashed_pw = hash_password(password)
        c.execute("INSERT INTO users (username, email, password) VALUES (?, ?, ?)",
                  (username, email, hashed_pw))
        
        conn.commit()
        conn.close()
        return True, "Registration successful!"
    except sqlite3.IntegrityError:
        return False, "Username already exists!"
    except Exception as e:
        return False, f"Error: {str(e)}"

def login_user(username, password):
    """Login user"""
    try:
        conn = sqlite3.connect('oil_spill_detection.db')
        c = conn.cursor()
        
        hashed_pw = hash_password(password)
        c.execute("SELECT id, username FROM users WHERE username=? AND password=?",
                  (username, hashed_pw))
        
        result = c.fetchone()
        conn.close()
        
        if result:
            return True, "Login successful!", result[0]
        else:
            return False, "Invalid username or password!", None
    except Exception as e:
        return False, f"Error: {str(e)}", None

def add_to_history(user_id, image_name, metrics, image_file):
    """Add detection to history with image data"""
    try:
        conn = sqlite3.connect('oil_spill_detection.db')
        c = conn.cursor()
        
        # Read image file and convert to binary
        image_file.seek(0)  # Reset file pointer to beginning
        image_binary = image_file.read()
        
        c.execute("""INSERT INTO history 
                     (user_id, image_name, image_data, oil_spill_area, confidence, severity, affected_area_km2)
                     VALUES (?, ?, ?, ?, ?, ?, ?)""",
                  (user_id, image_name, image_binary, metrics['oil_spill_area'], 
                   metrics['confidence'], metrics['severity'], metrics['affected_area_km2']))
        
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        st.error(f"Error saving to history: {e}")
        return False

def get_user_history(user_id):
    """Get user's detection history with images"""
    try:
        conn = sqlite3.connect('oil_spill_detection.db')
        c = conn.cursor()
        
        c.execute("""SELECT image_name, image_data, oil_spill_area, confidence, severity, 
                     affected_area_km2, timestamp 
                     FROM history WHERE user_id=? ORDER BY timestamp DESC""",
                  (user_id,))
        
        results = c.fetchall()
        conn.close()
        return results
    except Exception as e:
        st.error(f"Error fetching history: {e}")
        return []




RECEIVER_EMAIL = "madhu100luck@gmail.com"

def save_feedback(user_email, feedback_text):
    """Save feedback including user_id"""
    try:
        conn = sqlite3.connect('oil_spill_detection.db')
        c = conn.cursor()
        c.execute(
            "INSERT INTO feedback (user_id, username, feedback_text) VALUES (?, ?, ?)",
            (st.session_state.user_id, user_email, feedback_text)
        )
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        st.error(f"Error saving feedback: {e}")
        return False

# ==================== ML MODEL FUNCTIONS ====================

def dice_coef_improved(y_true, y_pred, smooth=1e-7):
    y_true_f = K.flatten(K.cast(y_true, 'float32'))
    y_pred_f = K.flatten(K.cast(y_pred > 0.5, 'float32'))
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def iou_improved(y_true, y_pred, smooth=1e-7):
    y_true_f = K.flatten(K.cast(y_true, 'float32'))
    y_pred_f = K.flatten(K.cast(y_pred > 0.5, 'float32'))
    intersection = K.sum(y_true_f * y_pred_f)
    union = K.sum(y_true_f) + K.sum(y_pred_f) - intersection
    return (intersection + smooth) / (union + smooth)

def precision_improved(y_true, y_pred, smooth=1e-7):
    y_pred_f = K.cast(y_pred > 0.5, 'float32')
    tp = K.sum(K.cast(y_true, 'float32') * y_pred_f)
    fp = K.sum((1 - K.cast(y_true, 'float32')) * y_pred_f)
    return (tp + smooth) / (tp + fp + smooth)

def recall_improved(y_true, y_pred, smooth=1e-7):
    y_pred_f = K.cast(y_pred > 0.5, 'float32')
    tp = K.sum(K.cast(y_true, 'float32') * y_pred_f)
    fn = K.sum(K.cast(y_true, 'float32') * (1 - y_pred_f))
    return (tp + smooth) / (tp + fn + smooth)

@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model(
            'best_oil_spill_model_v2.h5',
            custom_objects={
                'combined_loss_improved': lambda y_t, y_p: 0,
                'dice_coef_improved': dice_coef_improved,
                'iou_improved': iou_improved,
                'precision_improved': precision_improved,
                'recall_improved': recall_improved
            },
            compile=False
        )
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.info("Make sure 'best_oil_spill_model_v2.h5' is in the same directory")
        return None

def preprocess_image(image):
    img = np.array(image)
    
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    
    img_blur = cv2.GaussianBlur(img, (3, 3), 0)
    lab = cv2.cvtColor(img_blur, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    img_enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    img_enhanced = cv2.bilateralFilter(img_enhanced, 9, 75, 75)
    
    img_resized = cv2.resize(img_enhanced, IMG_SIZE)
    img_normalized = img_resized.astype(np.float32) / 127.5 - 1.0
    
    return img_normalized, img

def calculate_metrics(prediction, threshold=0.5):
    pred_binary = (prediction > threshold).astype(np.float32)
    
    total_pixels = prediction.size
    oil_spill_pixels = np.sum(pred_binary)
    clean_pixels = total_pixels - oil_spill_pixels
    oil_spill_percentage = (oil_spill_pixels / total_pixels) * 100
    
    if oil_spill_pixels > 0:
        confidence = np.mean(prediction[pred_binary == 1]) * 100
        max_confidence = np.max(prediction[pred_binary == 1]) * 100
        min_confidence = np.min(prediction[pred_binary == 1]) * 100
    else:
        confidence = 0
        max_confidence = 0
        min_confidence = 0
    
    if oil_spill_percentage > 15:
        severity = "CRITICAL"
    elif oil_spill_percentage > 5:
        severity = "HIGH"
    elif oil_spill_percentage > 1:
        severity = "MODERATE"
    elif oil_spill_percentage > 0.1:
        severity = "LOW"
    else:
        severity = "CLEAN"
    
    pixel_to_meters = 10
    affected_area_m2 = oil_spill_pixels * (pixel_to_meters ** 2)
    
    return {
        'oil_spill_area': oil_spill_percentage,
        'confidence': confidence,
        'max_confidence': max_confidence,
        'min_confidence': min_confidence,
        'total_pixels': total_pixels,
        'affected_pixels': int(oil_spill_pixels),
        'clean_pixels': int(clean_pixels),
        'severity': severity,
        'affected_area_m2': affected_area_m2,
        'affected_area_km2': affected_area_m2 / 1e6
    }

def create_visualizations(original_img, prediction, threshold=0.5):
    pred_squeeze = np.clip(prediction.squeeze(), 0, 1)
    pred_binary = (pred_squeeze > threshold).astype(np.uint8)
    
    original_resized = cv2.resize(original_img, (pred_squeeze.shape[1], pred_squeeze.shape[0]))
    if len(original_resized.shape) == 2:
        original_resized = cv2.cvtColor(original_resized, cv2.COLOR_GRAY2RGB)
    
    # --- Create combined figure as before ---
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(original_img)
    ax1.set_title('Original Image', fontsize=16, fontweight='bold', pad=10)
    ax1.axis('off')
    
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(original_resized)
    ax2.set_title('Processed Image (320x320)', fontsize=16, fontweight='bold', pad=10)
    ax2.axis('off')
    
    # Binary mask
    binary_display = 255 * np.ones_like(pred_binary, dtype=np.uint8)
    binary_display[pred_binary == 1] = 0
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.imshow(binary_display, cmap='gray', vmin=0, vmax=255)
    ax3.set_title('Binary Mask', fontsize=16, fontweight='bold', pad=10)
    ax3.axis('off')
    
    # Red overlay
    overlay_red = original_resized.copy()
    mask_rgb = np.zeros_like(overlay_red, dtype=np.uint8)
    mask_rgb[pred_binary == 1] = [255, 0, 0]
    overlay_result = cv2.addWeighted(overlay_red, 0.7, mask_rgb, 0.3, 0)
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.imshow(overlay_result)
    ax4.set_title('Red Overlay', fontsize=16, fontweight='bold', pad=10)
    ax4.axis('off')
    
    # Heatmap
    ax5 = fig.add_subplot(gs[1, 1])
    im = ax5.imshow(pred_squeeze, cmap='hot', vmin=0, vmax=1)
    ax5.set_title('Probability Heatmap', fontsize=16, fontweight='bold', pad=10)
    ax5.axis('off')
    cbar = plt.colorbar(im, ax=ax5, fraction=0.046, pad=0.04)
    cbar.set_label('Confidence', rotation=270, labelpad=20)
    
    # Contours
    contour_img = original_resized.copy()
    contours, _ = cv2.findContours(pred_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(contour_img, contours, -1, (0, 255, 0), 2)
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.imshow(contour_img)
    ax6.set_title(f'Contour Detection ({len(contours)} spill regions)', fontsize=16, fontweight='bold', pad=10)
    ax6.axis('off')
    
    plt.suptitle('Oil Spill Detection Analysis', fontsize=20, fontweight='bold', y=0.98)
    
    # Return all individual images too
    return fig, len(contours), binary_display, overlay_result, pred_squeeze


def fig_to_image(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    return buf


# ==================== PAGE FUNCTIONS ====================

def login_page():
    st.markdown('<h1 class="main-header">🛢️ Oil Spill Detection System</h1>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown('<div class="login-box">', unsafe_allow_html=True)
        
        tab1, tab2 = st.tabs(["🔐 Login", "📝 Register"])
        
        with tab1:
            st.subheader("Login to Your Account")
            login_username = st.text_input("Username", key="login_user")
            login_password = st.text_input("Password", type="password", key="login_pass")
            
            if st.button("Login", type="primary", use_container_width=True):
                if login_username and login_password:
                    success, message, user_id = login_user(login_username, login_password)
                    if success:
                        st.session_state.logged_in = True
                        st.session_state.username = login_username
                        st.session_state.user_id = user_id
                        st.session_state.page = 'detection'
                        st.success(message)
                        st.rerun()
                    else:
                        st.error(message)
                else:
                    st.warning("Please enter both username and password")
        
        with tab2:
            st.subheader("Create New Account")
            reg_username = st.text_input("Username", key="reg_user")
            reg_email = st.text_input("Email", key="reg_email")
            reg_password = st.text_input("Password", type="password", key="reg_pass")
            reg_password2 = st.text_input("Confirm Password", type="password", key="reg_pass2")
            
            if st.button("Register", type="primary", use_container_width=True):
                if reg_username and reg_email and reg_password and reg_password2:
                    if reg_password == reg_password2:
                        if len(reg_password) >= 6:
                            success, message = register_user(reg_username, reg_email, reg_password)
                            if success:
                                st.success(message + " Please login now.")
                            else:
                                st.error(message)
                        else:
                            st.error("Password must be at least 6 characters")
                    else:
                        st.error("Passwords do not match")
                else:
                    st.warning("Please fill in all fields")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("---")
        st.info("🔬 Advanced AI-powered oil spill detection from satellite imagery")


def detection_page():
    # Header with navigation
    col_h1, col_h2, col_h3, col_h4 = st.columns([3, 1, 1, 1])
    
    with col_h1:
        st.markdown('<h1 class="main-header">🛢️ Oil Spill Detection</h1>', unsafe_allow_html=True)
    
    with col_h2:
        if st.button("📜 History", use_container_width=True):
            st.session_state.page = 'history'
            st.rerun()
    
    with col_h3:
        if st.button("💬 Feedback", use_container_width=True):
            st.session_state.page = 'feedback'
            st.rerun()
    
    with col_h4:
        if st.button("🚪 Logout", use_container_width=True):
            st.session_state.logged_in = False
            st.session_state.username = None
            st.session_state.user_id = None
            st.session_state.page = 'login'
            st.rerun()
    
    if st.session_state.username == "admin":
        with col_h4:
            if st.button("🛠️ Admin Panel", use_container_width=True):
                st.session_state.page = 'admin'
                st.rerun()
    
    st.markdown(f"**Welcome, {st.session_state.username}!** 👋")
    st.markdown("---")
    
    model = load_model()
    if model is None:
        st.stop()
    st.success("✅ Model loaded successfully!")
    
    # Sidebar with threshold
    with st.sidebar:
        st.header("⚙️ Detection Settings")
        threshold = st.slider(
            "Detection Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.95,
            step=0.05,
            help="Lower = more sensitive, Higher = more conservative"
        )
        st.markdown("---")
        st.markdown("### 📊 About")
        st.info(
            "This AI system detects oil spills in satellite/aerial imagery using "
            "deep learning. Upload an image to get comprehensive analysis."
        )
        st.markdown("### 🎯 Model Info")
        st.write("• **Architecture:** Deep Attention U-Net")
        st.write("• **Input Size:** 320×320 pixels")
        st.write("• **Preprocessing:** CLAHE + Bilateral Filter")
    
    # File uploader
    st.markdown('<p class="upload-text">📤 Upload satellite/aerial image for oil spill detection</p>', 
                unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Choose an image...",
        type=['png', 'jpg', 'jpeg', 'tiff', 'bmp'],
        help="Supported formats: PNG, JPG, JPEG, TIFF, BMP"
    )
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("📷 Uploaded Image")
            st.image(image, use_container_width=True)
            st.caption(f"Image size: {image.size[0]} × {image.size[1]} pixels")
        
        with st.spinner('🔍 Analyzing image for oil spills...'):
            preprocessed_img, original_img = preprocess_image(image)
            input_img = np.expand_dims(preprocessed_img, axis=0)
            prediction = model.predict(input_img, verbose=0)[0]
            metrics = calculate_metrics(prediction, threshold)
            # Save to history WITH image file
            add_to_history(st.session_state.user_id, uploaded_file.name, metrics, uploaded_file)
        
        with col2:
            st.subheader("📊 Detection Summary")
            
            if metrics['severity'] == "CRITICAL":
                st.markdown('<div class="alert-danger">🚨 CRITICAL: Severe Oil Spill Detected!</div>', 
                           unsafe_allow_html=True)
            elif metrics['severity'] == "HIGH":
                st.markdown('<div class="alert-danger">⚠️ HIGH SEVERITY: Major Oil Spill Detected!</div>', 
                           unsafe_allow_html=True)
            elif metrics['severity'] == "MODERATE":
                st.markdown('<div class="alert-warning">⚠️ MODERATE: Oil Spill Detected</div>', 
                           unsafe_allow_html=True)
            elif metrics['severity'] == "LOW":
                st.markdown('<div class="alert-warning">⚠️ LOW: Minor Oil Spill Detected</div>', 
                           unsafe_allow_html=True)
            else:
                st.markdown('<div class="alert-success">✅ CLEAN: No Significant Oil Spill</div>', 
                           unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            metric_col1, metric_col2 = st.columns(2)
            with metric_col1:
                st.metric("🛢️ Oil Spill Coverage", f"{metrics['oil_spill_area']:.2f}%")
                st.metric("🎯 Avg Confidence", f"{metrics['confidence']:.1f}%")
                st.metric("📍 Affected Pixels", f"{metrics['affected_pixels']:,}")
            
            with metric_col2:
                st.metric("📏 Estimated Area", f"{metrics['affected_area_km2']:.3f} km²")
                st.metric("📊 Max Confidence", f"{metrics['max_confidence']:.1f}%")
                st.metric("✅ Clean Pixels", f"{metrics['clean_pixels']:,}")
        
        st.markdown("---")
        st.subheader("🎨 Detection Visualizations")
        
        # Create visualizations
        fig, num_contours, binary_img, overlay_img, heatmap_img = create_visualizations(original_img, prediction, threshold)
        st.pyplot(fig)
        st.info(f"🔍 Detected **{num_contours}** separate oil spill region(s)")
        
        st.markdown("---")
        st.subheader("💾 Download Results")

        # Download buttons for various visualizations
        img_buf = fig_to_image(fig)
        st.download_button(
            label="📥 Download All Visualizations",
            data=img_buf,
            file_name=f"oil_spill_analysis_{uploaded_file.name}.png",
            mime="image/png",
            use_container_width=True
        )
        plt.close(fig)

        # Binary Mask
        fig_bin, ax_bin = plt.subplots()
        ax_bin.imshow(binary_img, cmap='gray', vmin=0, vmax=255)
        ax_bin.axis('off')
        buf_bin = fig_to_image(fig_bin)
        st.download_button(
            label="📥 Download Binary Mask",
            data=buf_bin,
            file_name=f"binary_mask_{uploaded_file.name}.png",
            mime="image/png",
            use_container_width=True
        )
        plt.close(fig_bin)

        # Red Overlay
        fig_overlay, ax_overlay = plt.subplots()
        ax_overlay.imshow(overlay_img)
        ax_overlay.axis('off')
        buf_overlay = fig_to_image(fig_overlay)
        st.download_button(
            label="📥 Download Red Overlay",
            data=buf_overlay,
            file_name=f"red_overlay_{uploaded_file.name}.png",
            mime="image/png",
            use_container_width=True
        )
        plt.close(fig_overlay)

        # Probability Heatmap
        fig_heat, ax_heat = plt.subplots()
        im = ax_heat.imshow(heatmap_img, cmap='hot', vmin=0, vmax=1)
        ax_heat.axis('off')
        cbar = plt.colorbar(im, ax=ax_heat, fraction=0.046, pad=0.04)
        cbar.set_label('Confidence', rotation=270, labelpad=15)
        buf_heat = fig_to_image(fig_heat)
        st.download_button(
            label="📥 Download Probability Heatmap",
            data=buf_heat,
            file_name=f"heatmap_{uploaded_file.name}.png",
            mime="image/png",
            use_container_width=True
        )
        plt.close(fig_heat)
    else:
        st.info("👆 Please upload an image to start detection")

def history_page():
    col_h1, col_h2 = st.columns([4, 1])
    
    with col_h1:
        st.markdown('<h1 class="main-header">📜 Detection History</h1>', unsafe_allow_html=True)
    
    with col_h2:
        if st.button("🔙 Back", use_container_width=True):
            st.session_state.page = 'detection'
            st.rerun()
    
    st.markdown(f"**User:** {st.session_state.username}")
    st.markdown("---")
    
    history = get_user_history(st.session_state.user_id)
    
    if not history:
        st.info("📭 No detection history yet. Upload images to start building your history!")
    else:
        st.success(f"📊 Total detections: **{len(history)}**")
        st.markdown("---")
        
        for idx, entry in enumerate(history, 1):
            st.markdown(f'<div class="history-card">', unsafe_allow_html=True)
            
            # Create columns for image and details
            col_img, col_details = st.columns([1, 3])
            
            with col_img:
                # Display the stored image
                if entry[1]:  # image_data exists
                    try:
                        image = Image.open(io.BytesIO(entry[1]))
                        st.image(image, use_container_width=True, caption=entry[0])
                    except Exception as e:
                        st.write(f"🖼️ **{entry[0]}**")
                        st.caption("Image preview unavailable")
                else:
                    st.write(f"🖼️ **{entry[0]}**")
            
            with col_details:
                st.write(f"**📅 Date:** {entry[6]}")
                
                # Create metric columns
                metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
                
                with metric_col1:
                    st.metric("Coverage", f"{entry[2]:.2f}%")
                
                with metric_col2:
                    st.metric("Confidence", f"{entry[3]:.1f}%")
                
                with metric_col3:
                    st.metric("Area", f"{entry[5]:.3f} km²")
                
                with metric_col4:
                    severity_emoji = {
                        'CRITICAL': '🔴',
                        'HIGH': '🟠',
                        'MODERATE': '🟡',
                        'LOW': '🟢',
                        'CLEAN': '✅'
                    }
                    st.metric("Severity", f"{severity_emoji.get(entry[4], '')} {entry[4]}")
            
            st.markdown('</div>', unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
def feedback_page():
    import pytz
    
    col_h1, col_h2 = st.columns([4, 1])
    
    with col_h1:
        st.markdown('<h1 class="main-header">💬 Send Feedback</h1>', unsafe_allow_html=True)
    
    with col_h2:
        if st.button("🔙 Back", use_container_width=True):
            st.session_state.page = 'detection'
            st.rerun()  # updated from experimental_rerun
    
    st.markdown(f"**User:** {st.session_state.username}")
    st.markdown("---")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.subheader("📝 We'd love to hear from you!")
        
        feedback_text = st.text_area(
            "Your Feedback",
            height=200,
            placeholder="Share your thoughts, suggestions, or report issues..."
        )
        user_email = st.text_input("Your email (optional)")
        
        # Track if feedback button was clicked
        if 'feedback_clicked' not in st.session_state:
            st.session_state.feedback_clicked = False

        if st.button("📤 Generate Email Link", type="primary", use_container_width=True):
            if feedback_text.strip():
                st.session_state.feedback_clicked = True
                
                # Save feedback to database
                save_feedback(user_email or st.session_state.username, feedback_text)
                
                # Indian Standard Time
                ist = pytz.timezone('Asia/Kolkata')
                now_ist = datetime.now(ist).strftime("%Y-%m-%d %H:%M:%S IST")
                
                subject = f"Feedback from {user_email or st.session_state.username or 'Anonymous'}"
                body_lines = [
                    f"Time: {now_ist}",
                    f"User email: {user_email or 'Not provided'}",
                    "",
                    "---- Feedback ----",
                    feedback_text
                ]
                body = "\n".join(body_lines)
                
                # URL encode
                params = {"subject": subject, "body": body}
                query = urllib.parse.urlencode(params, quote_via=urllib.parse.quote)
                mailto_link = f"mailto:{RECEIVER_EMAIL}?{query}"
                
                # Show mailto link and balloons
                st.markdown(
                    f"[✉️ Click here to open your email client and send feedback]({mailto_link})",
                    unsafe_allow_html=True
                )
                st.success("✅ Your feedback has been saved and is ready to send via your email client!")
                st.balloons()
                
            else:
                st.warning("Please enter your feedback before submitting")
        
        st.markdown("---")
        st.info("💡 Your feedback helps us improve the Oil Spill Detection System. Thank you!")



def admin_panel():
    col_h1, col_h2 = st.columns([4, 1])
    
    with col_h1:
        st.markdown('<h1 class="main-header">🛠️ Admin Panel</h1>', unsafe_allow_html=True)
    
    with col_h2:
        if st.button("🔙 Back", use_container_width=True):
            st.session_state.page = 'detection'
            st.rerun()
    
    option = st.sidebar.selectbox("Choose view", ["Users", "Detection History", "Feedback"])
    
    conn = sqlite3.connect('oil_spill_detection.db')
    c = conn.cursor()
    
    if option == "Users":
        st.subheader("👥 Registered Users")
        c.execute("SELECT id, username, email, created_at FROM users ORDER BY created_at DESC")
        users = c.fetchall()
        if users:
            for user in users:
                st.markdown(f"**ID:** {user[0]} | **Username:** {user[1]} | **Email:** {user[2]} | **Registered On:** {user[3]}")
                st.markdown("---")
        else:
            st.info("No registered users found.")

    elif option == "Detection History":
        st.subheader("📊 Detection History")
        user_filter = st.text_input("Filter by username (optional)")
        if user_filter:
            c.execute("""
                SELECT h.id, u.username, h.image_name, h.oil_spill_area, h.confidence, h.severity, h.affected_area_km2, h.timestamp
                FROM history h
                JOIN users u ON h.user_id = u.id
                WHERE u.username LIKE ?
                ORDER BY h.timestamp DESC
            """, (f"%{user_filter}%",))
        else:
            c.execute("""
                SELECT h.id, u.username, h.image_name, h.oil_spill_area, h.confidence, h.severity, h.affected_area_km2, h.timestamp
                FROM history h
                JOIN users u ON h.user_id = u.id
                ORDER BY h.timestamp DESC
            """)
        history = c.fetchall()
        if history:
            for record in history:
                st.markdown(f"**ID:** {record[0]} | **User:** {record[1]} | **File:** {record[2]} | "
                            f"Coverage: {record[3]:.2f}% | Confidence: {record[4]:.1f}% | "
                            f"Severity: {record[5]} | Area: {record[6]:.3f} km² | Time: {record[7]}")
                st.markdown("---")
        else:
            st.info("No detection history found.")
    
    elif option == "Feedback":
        st.subheader("💬 User Feedback")
        c.execute("SELECT id, user_id, username, feedback_text, timestamp FROM feedback ORDER BY timestamp DESC")
        feedbacks = c.fetchall()
        if feedbacks:
            for fb in feedbacks:
                st.markdown(f"**ID:** {fb[0]} | **User ID:** {fb[1]} | **Username:** {fb[2]} | **Time:** {fb[4]}")
                st.markdown(f"**Feedback:** {fb[3]}")
                st.markdown("---")
        else:
            st.info("No feedback submitted yet.")


    elif option == "Detection History":
        st.subheader("📊 Detection History")
        user_filter = st.text_input("Filter by username (optional)")
        if user_filter:
            c.execute("""
                SELECT h.id, u.username, h.image_name, h.oil_spill_area, h.confidence, h.severity, h.affected_area_km2, h.timestamp
                FROM history h
                JOIN users u ON h.user_id = u.id
                WHERE u.username LIKE ?
                ORDER BY h.timestamp DESC
            """, (f"%{user_filter}%",))
        else:
            c.execute("""
                SELECT h.id, u.username, h.image_name, h.oil_spill_area, h.confidence, h.severity, h.affected_area_km2, h.timestamp
                FROM history h
                JOIN users u ON h.user_id = u.id
                ORDER BY h.timestamp DESC
            """)
        history = c.fetchall()
        if history:
            for record in history:
                st.markdown(f"**ID:** {record[0]} | **User:** {record[1]} | **File:** {record[2]} | "
                            f"Coverage: {record[3]:.2f}% | Confidence: {record[4]:.1f}% | "
                            f"Severity: {record[5]} | Area: {record[6]:.3f} km² | Time: {record[7]}")
                st.markdown("---")
        else:
            st.info("No detection history found.")
    
    elif option == "Feedback":
        st.subheader("💬 User Feedback")
        c.execute("SELECT id, user_id, username, feedback_text, timestamp FROM feedback ORDER BY timestamp DESC")
        feedbacks = c.fetchall()
        if feedbacks:
            for fb in feedbacks:
                st.markdown(f"**ID:** {fb[0]} | **User ID:** {fb[1]} | **Username:** {fb[2]} | **Time:** {fb[4]}")
                st.markdown(f"**Feedback:** {fb[3]}")
                st.markdown("---")
        else:
            st.info("No feedback submitted yet.")
    
    conn.close()



# ==================== MAIN APP ====================

def main():
    # Initialize database
    init_db()
    
    if not st.session_state.logged_in:
        login_page()
    else:
        if st.session_state.page == 'detection':
            detection_page()
        elif st.session_state.page == 'history':
            history_page()
        elif st.session_state.page == 'feedback':
            feedback_page()
        elif st.session_state.page == 'admin':
            admin_panel()

if __name__ == "__main__":
    main()



# Footer CSS + HTML (sticky)
st.markdown("""
<style>
.footer {
    position: fixed;
    
    left: 0;
    width: 100%;
    background-color: #0d47a1;  /* change as you like */
    color: white;
    text-align: center;
    padding: 10px 0;
    z-index: 100;
    font-family: 'Arial', sans-serif;
}
.footer-text-main {
    font-weight: bold;
    font-size: 16px;
    margin: 0;
}
.footer-text {
    font-size: 12px;
    margin: 0;
}
</style>

<div class="footer">
    <p class="footer-text-main">🌍 Oil Spill Detection Project</p>
    <p class="footer-text">Developed by Madhu Mitha | Infosys SpringBoard Virtual Internship</p>
    <p class="footer-text">© 2025 All Rights Reserved</p>
</div>
""", unsafe_allow_html=True)
