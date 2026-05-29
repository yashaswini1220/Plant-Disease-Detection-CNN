import streamlit as st
import numpy as np
import cv2

from PIL import Image
from tensorflow.keras.models import load_model

# =========================================
# PAGE CONFIG
# =========================================

st.set_page_config(
    page_title="Plant Disease Detection",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)


st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Playfair+Display:wght@600&display=swap');

/* ===== GLOBAL TYPOGRAPHY OVERRIDES ===== */
/* This forces ALL titles and headings to be pure white */
h1, h2, h3, h4, h5, h6 {
    color: #ffffff !important;
}

/* ===== ROOT VARIABLES ===== */
:root {
    --primary: #10b981;
    --primary-dark: #059669;
    --primary-light: #34d399;
    --accent: #f0fdf4;
    --dark: #064e3b;
    --darker: #022c22;
    --glass: rgba(0, 0, 0, 0.65);
    --glass-border: rgba(255, 255, 255, 0.2);
    --shadow: 0 8px 32px rgba(0, 0, 0, 0.6);
    --radius: 24px;
}

/* =========================
   FULL SCREEN BACKGROUND
========================= */

[data-testid="stAppViewContainer"] {
    background: black;
}

[data-testid="stAppViewContainer"]::before {
    content: "";
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;

    background-image: url("https://images.unsplash.com/photo-1500382017468-9049fed747ef");

    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;

    filter: blur(10px);
    transform: scale(1.1);

    z-index: -2;
}

[data-testid="stAppViewContainer"]::after {
    content: "";
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;

    background: rgba(0, 0, 0, 0.55);

    z-index: -1;
}

[data-testid="stHeader"] {
    background: transparent;
}

/* ===== SIDEBAR ===== */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #022c22 0%, #064e3b 100%);
    border-right: 1px solid var(--glass-border);
}

[data-testid="stSidebar"] .stMarkdown {
    color: #ffffff;
}

[data-testid="stSidebar"] h1, 
[data-testid="stSidebar"] h2, 
[data-testid="stSidebar"] h3 {
    color: #ffffff !important;
    font-family: 'Playfair Display', serif;
}

/* ===== TYPOGRAPHY ===== */
.hero-section {
    text-align: center;
    padding: 60px 20px 40px;
}

.hero-title {
    font-family: 'Playfair Display', serif;
    font-size: clamp(36px, 6vw, 64px);
    font-weight: 600;
    color: #ffffff !important;
    margin-bottom: 12px;
    letter-spacing: -0.02em;
    text-shadow: 0 4px 24px rgba(0, 0, 0, 0.8);
}

.hero-subtitle {
    font-size: clamp(16px, 2.5vw, 20px);
    font-weight: 300;
    color: #ffffff;
    max-width: 600px;
    margin: 0 auto;
    line-height: 1.6;
}

.hero-icon {
    font-size: 80px;
    margin-bottom: 20px;
    filter: drop-shadow(0 8px 16px rgba(16, 185, 129, 0.4));
}

/* ===== GLASS CARDS ===== */
.glass-card {
    background: var(--glass);
    backdrop-filter: blur(40px) !important; /* Increased Blur */
    -webkit-backdrop-filter: blur(40px) !important;
    border: 1px solid var(--glass-border);
    border-radius: var(--radius);
    padding: 32px;
    box-shadow: var(--shadow);
    transition: transform 0.3s ease, box-shadow 0.3s ease;
}

.glass-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 12px 40px rgba(0, 0, 0, 0.7);
}

/* ===== UPLOAD SECTION ===== */
.upload-section {
    background: rgba(0, 0, 0, 0.7);
    backdrop-filter: blur(40px) !important; /* Increased Blur */
    -webkit-backdrop-filter: blur(40px) !important;
    border: 2px dashed rgba(16, 185, 129, 0.6);
    border-radius: var(--radius);
    padding: 48px 32px;
    text-align: center;
    transition: all 0.3s ease;
}

.upload-section:hover {
    border-color: var(--primary);
    background: rgba(16, 185, 129, 0.25);
}

.upload-icon {
    font-size: 48px;
    margin-bottom: 16px;
}

.upload-text {
    color: #ffffff;
    font-size: 18px;
    font-weight: 600;
    margin-bottom: 8px;
}

.upload-hint {
    color: #ffffff;
    font-size: 14px;
    opacity: 0.9;
}

/* ===== RESULT CARDS ===== */
.result-card {
    background: linear-gradient(135deg, rgba(16, 185, 129, 0.3) 0%, rgba(5, 150, 105, 0.4) 100%);
    backdrop-filter: blur(40px) !important; /* Increased Blur */
    -webkit-backdrop-filter: blur(40px) !important;
    border: 1px solid rgba(16, 185, 129, 0.6);
    border-radius: var(--radius);
    padding: 40px;
    text-align: center;
    box-shadow: var(--shadow), 0 0 60px rgba(16, 185, 129, 0.2);
}

.result-label {
    font-size: 14px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #ffffff;
    margin-bottom: 12px;
}

.result-disease {
    font-family: 'Playfair Display', serif;
    font-size: clamp(24px, 4vw, 36px);
    font-weight: 600;
    color: #ffffff !important;
    margin-bottom: 20px;
    line-height: 1.3;
}

.confidence-container {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 16px;
    margin-top: 24px;
}

.confidence-badge {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background: rgba(16, 185, 129, 0.5);
    padding: 12px 24px;
    border-radius: 50px;
    font-size: 18px;
    font-weight: 600;
    color: #ffffff;
}

/* ===== WARNING CARD ===== */
.warning-card {
    background: linear-gradient(135deg, rgba(239, 68, 68, 0.4) 0%, rgba(185, 28, 28, 0.5) 100%);
    backdrop-filter: blur(40px) !important; /* Increased Blur */
    -webkit-backdrop-filter: blur(40px) !important;
    border: 1px solid rgba(239, 68, 68, 0.6);
    border-radius: var(--radius);
    padding: 32px;
    text-align: center;
    box-shadow: var(--shadow);
}

.warning-icon {
    font-size: 48px;
    margin-bottom: 16px;
}

.warning-title {
    font-size: 20px;
    font-weight: 600;
    color: #ffffff !important;
    margin-bottom: 8px;
}

.warning-text {
    color: #ffffff;
    font-size: 16px;
    line-height: 1.6;
}

/* ===== INFO CARDS ===== */
.info-section {
    margin-top: 32px;
}

.info-card {
    background: var(--glass);
    backdrop-filter: blur(40px) !important; /* Increased Blur */
    -webkit-backdrop-filter: blur(40px) !important;
    border: 1px solid var(--glass-border);
    border-radius: var(--radius);
    padding: 32px;
    box-shadow: var(--shadow);
}

.info-header {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 20px;
}

.info-icon {
    font-size: 28px;
}

.info-title {
    font-family: 'Playfair Display', serif;
    font-size: 22px;
    font-weight: 600;
    color: #ffffff !important;
    margin: 0;
}

.info-text {
    color: #ffffff;
    font-size: 16px;
    line-height: 1.8;
}

.tips-list {
    list-style: none;
    padding: 0;
    margin: 0;
}

.tips-list li {
    display: flex;
    align-items: flex-start;
    gap: 12px;
    color: #ffffff;
    font-size: 16px;
    line-height: 1.6;
    padding: 12px 0;
    border-bottom: 1px solid rgba(255, 255, 255, 0.15);
}

.tips-list li:last-child {
    border-bottom: none;
}

.tip-bullet {
    color: var(--primary-light);
    font-size: 20px;
    flex-shrink: 0;
}

/* ===== IMAGE DISPLAY ===== */
.image-container {
    background: var(--glass);
    backdrop-filter: blur(40px) !important; /* Increased Blur */
    -webkit-backdrop-filter: blur(40px) !important;
    border: 1px solid var(--glass-border);
    border-radius: var(--radius);
    padding: 24px;
    text-align: center;
    box-shadow: var(--shadow);
}

.image-container img {
    border-radius: 16px;
    max-width: 100%;
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.5);
}

.image-label {
    color: #ffffff;
    font-size: 14px;
    font-weight: 600;
    margin-top: 16px;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

/* ===== FOOTER ===== */
.footer {
    text-align: center;
    padding: 48px 20px 32px;
    color: #ffffff;
    font-size: 14px;
}

.footer hr {
    border: none;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.4), transparent);
    margin-bottom: 24px;
}

.footer-tech {
    display: flex;
    justify-content: center;
    gap: 24px;
    flex-wrap: wrap;
    margin-top: 16px;
}

.tech-badge {
    background: rgba(0, 0, 0, 0.5);
    padding: 8px 16px;
    border-radius: 50px;
    font-size: 12px;
    color: #ffffff;
    border: 1px solid var(--glass-border);
}

/* ===== STREAMLIT OVERRIDES ===== */
.stFileUploader > div > div {
    background: transparent !important;
}

.stFileUploader label {
    color: #ffffff !important;
    font-weight: 600;
}

div[data-testid="stFileUploadDropzone"] {
    background: rgba(16, 185, 129, 0.2) !important;
    border: 2px dashed rgba(16, 185, 129, 0.6) !important;
    border-radius: 16px !important;
}

div[data-testid="stFileUploadDropzone"]:hover {
    border-color: var(--primary) !important;
    background: rgba(16, 185, 129, 0.3) !important;
}

.stSpinner > div {
    border-color: var(--primary) transparent transparent transparent !important;
}

</style>
""", unsafe_allow_html=True)

# =========================================
# LOAD MODEL
# =========================================

@st.cache_resource
def load_disease_model():
    return load_model("plant_disease_model.keras")

model = load_disease_model()

# =========================================
# LOAD CLASS NAMES
# =========================================

@st.cache_data
def load_class_names():
    return np.load("class_names.npy", allow_pickle=True)

class_names = load_class_names()

# =========================================
# DISEASE DATA
# =========================================

disease_data = {
    "Pepper_Bacterial_Spot": {
        "precaution": "Remove infected leaves immediately and avoid overhead watering to prevent spread.",
        "tips": [
            "Apply copper-based fungicides weekly",
            "Maintain proper plant spacing for airflow",
            "Avoid working with wet plants"
        ]
    },
    "Pepper_Healthy": {
        "precaution": "Your plant is thriving! Continue with current care practices.",
        "tips": [
            "Maintain consistent watering schedule",
            "Apply balanced fertilizer monthly",
            "Monitor leaves weekly for early detection"
        ]
    },
    "Potato_Early_Blight": {
        "precaution": "Apply fungicides promptly and remove all infected foliage.",
        "tips": [
            "Practice crop rotation annually",
            "Water at soil level to keep foliage dry",
            "Improve air circulation between plants"
        ]
    },
    "Potato_Healthy": {
        "precaution": "Excellent! Your potato plant shows no signs of disease.",
        "tips": [
            "Maintain proper irrigation depth",
            "Add organic compost for nutrients",
            "Scout regularly for pest activity"
        ]
    },
    "Potato_Late_Blight": {
        "precaution": "Critical: Remove and destroy infected plants immediately to prevent spread.",
        "tips": [
            "Eliminate standing water around plants",
            "Apply systemic fungicides preventively",
            "Improve soil drainage"
        ]
    },
    "Tomato_Bacterial_Spot": {
        "precaution": "Use only certified disease-free seeds and avoid wetting foliage.",
        "tips": [
            "Sanitize tools between plants",
            "Apply copper sprays preventively",
            "Avoid overcrowding plants"
        ]
    },
    "Tomato_Early_Blight": {
        "precaution": "Remove and dispose of infected leaves immediately.",
        "tips": [
            "Rotate tomato placement yearly",
            "Apply fungicides at first sign",
            "Ensure good air circulation"
        ]
    },
    "Tomato_Healthy": {
        "precaution": "Perfect! Your tomato plant is in excellent condition.",
        "tips": [
            "Continue consistent deep watering",
            "Side-dress with balanced fertilizer",
            "Stake or cage for optimal growth"
        ]
    },
    "Tomato_Late_Blight": {
        "precaution": "Reduce humidity levels and improve air circulation immediately.",
        "tips": [
            "Apply fungicides preventively",
            "Avoid overhead irrigation",
            "Remove lower leaves for airflow"
        ]
    },
    "Tomato_Leaf_Mold": {
        "precaution": "Reduce greenhouse humidity and increase ventilation.",
        "tips": [
            "Install fans for air movement",
            "Space plants adequately",
            "Consider resistant varieties"
        ]
    },
    "Tomato_Septoria_Leaf_Spot": {
        "precaution": "Remove infected foliage and destroy—do not compost.",
        "tips": [
            "Use drip irrigation to avoid splashing",
            "Apply mulch to prevent soil contact",
            "Maintain garden sanitation"
        ]
    },
    "Tomato_Spider_Mites": {
        "precaution": "Apply insecticidal soap or neem oil immediately.",
        "tips": [
            "Check leaf undersides regularly",
            "Increase humidity around plants",
            "Introduce beneficial predatory mites"
        ]
    },
    "Tomato_Target_Spot": {
        "precaution": "Apply appropriate fungicides and improve growing conditions.",
        "tips": [
            "Increase plant spacing",
            "Prune for better airflow",
            "Remove fallen debris"
        ]
    },
    "Tomato_Mosaic_Virus": {
        "precaution": "Remove and destroy infected plants—there is no cure.",
        "tips": [
            "Disinfect all tools with bleach solution",
            "Control aphid populations",
            "Plant resistant varieties"
        ]
    },
    "Tomato_Yellow_Leaf_Curl_Virus": {
        "precaution": "Control whitefly populations immediately to prevent spread.",
        "tips": [
            "Use resistant tomato varieties",
            "Install yellow sticky traps",
            "Apply insecticidal treatments"
        ]
    }
}

# =========================================
# HELPER FUNCTIONS
# =========================================

def normalize_disease_name(name):
    """Clean up inconsistent naming from dataset labels."""
    while "__" in name:
        name = name.replace("__", "_")
    
    replacements = {
        "healthy": "Healthy",
        "Early_blight": "Early_Blight",
        "Late_blight": "Late_Blight",
        "Spider_mites": "Spider_Mites",
        "Septoria_leaf_spot": "Septoria_Leaf_Spot",
        "Mosaic_virus": "Mosaic_Virus",
        "YellowLeafCurlVirus": "Yellow_Leaf_Curl_Virus"
    }
    
    for old, new in replacements.items():
        name = name.replace(old, new)
    
    return name

def format_display_name(name):
    """Convert internal name to human-readable format."""
    return name.replace("_", " ")

# =========================================
# SIDEBAR
# =========================================

with st.sidebar:
    st.markdown("## 🌿 About")
    st.markdown("""
    **Plant Disease Detection** uses deep learning to identify diseases in plant leaves from uploaded images.
    
    ---
    
    **Technology Stack:**
    - 🧠 CNN Deep Learning
    - 📊 TensorFlow & Keras
    - 🖼️ OpenCV Processing
    - 🎨 Streamlit Interface
    
    ---
    
    **How to use:**
    1. Upload a clear leaf image
    2. Wait for analysis
    3. Review diagnosis & tips
    
    ---
    
    **Supported Plants:**
    - 🫑 Pepper
    - 🥔 Potato
    - 🍅 Tomato
    """)

# =========================================
# HERO SECTION
# =========================================

st.markdown("""
<div class="hero-section">
    <div class="hero-icon">🌿</div>
    <h1 class="hero-title">Plant Disease Detection</h1>
</div>
""", unsafe_allow_html=True)

# =========================================
# MAIN CONTENT
# =========================================

col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    st.markdown("""
    <div class="upload-section">
        <div class="upload-icon">📤</div>
        <p class="upload-text">Drop your leaf image here</p>
        <p class="upload-hint">Supports JPG, JPEG, PNG</p>
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Upload Plant Leaf",
        type=["jpg", "jpeg", "png"],
        key="leaf_uploader",
        label_visibility="collapsed"
    )

# =========================================
# PREDICTION & RESULTS
# =========================================

if uploaded_file is not None:
    
    # Create two columns for image and result
    img_col, result_col = st.columns([1, 1], gap="large")
    
    with img_col:
        image = Image.open(uploaded_file)
        
        st.markdown("""
        <div class="image-container">
        """, unsafe_allow_html=True)
        
        st.image(image, use_container_width=True)
        
        st.markdown("""
            <p class="image-label">📷 Uploaded Specimen</p>
        </div>
        """, unsafe_allow_html=True)
    
    with result_col:
        with st.spinner("Analyzing leaf pattern..."):
            # Preprocess image
            img = np.array(image)
            
            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            elif img.shape[2] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
            
            img = cv2.resize(img, (128, 128))
            img = img / 255.0
            img = np.reshape(img, (1, 128, 128, 3))
            
            # Predict
            prediction = model.predict(img, verbose=0)
            predicted_class = np.argmax(prediction)
            confidence = np.max(prediction) * 100
            # Get top 3 predictions
            top3_idx = np.argsort(prediction[0])[-3:][::-1]
            print("\nTop 3 Predictions:")
            for i in top3_idx:
                print(f"{class_names[i]} : {prediction[0][i] * 100:.2f}%")
                predicted_class = np.argmax(prediction)
                confidence = np.max(prediction) * 100
            # Get and clean disease name
            disease_name = normalize_disease_name(class_names[predicted_class])
            display_name = format_display_name(disease_name)
            # Top 3 Predictions Display
            st.markdown("### 🔎 Top 3 Predictions")
            for i in top3_idx:
                prob = prediction[0][i] * 100
                st.markdown(
                    f"""
                    <p style='color:white;
                  font-size:18px;
                  font-weight:600;'>
                  {class_names[i].replace('_',' ')} : {prob:.2f}%
                  </p>
                  """,
                  unsafe_allow_html=True
                  )
                st.progress(float(prediction[0][i]))
        
        # Display results based on confidence
        if confidence < 45:
            st.markdown(f"""
            <div class="warning-card">
                <div class="warning-icon">⚠️</div>
                <p class="warning-title">Low Confidence Detection</p>
                <p class="warning-text">
                    The model couldn't identify the disease with sufficient certainty 
                    ({confidence:.1f}% confidence). Please upload a clearer, well-lit 
                    image of the leaf.
                </p>
            </div>
            """, unsafe_allow_html=True)
        else:
            # Determine if healthy
            is_healthy = "Healthy" in disease_name
            status_emoji = "✨" if is_healthy else "🔬"
            status_label = "HEALTHY PLANT" if is_healthy else "DISEASE DETECTED"
            
            st.markdown(f"""
            <div class="result-card">
                <p class="result-label">{status_label}</p>
                <h2 class="result-disease">{status_emoji} {display_name}</h2>
                <div class="confidence-container">
                    <span class="confidence-badge">
                        🎯 {confidence:.1f}% Confidence
                    </span>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    # Show precautions and tips if confident prediction
    if confidence >= 40 and disease_name in disease_data:
        st.markdown("<div class='info-section'>", unsafe_allow_html=True)
        
        info_col1, info_col2 = st.columns(2, gap="large")
        
        data = disease_data[disease_name]
        
        with info_col1:
            st.markdown(f"""
            <div class="info-card">
                <div class="info-header">
                    <span class="info-icon">🛡️</span>
                    <h3 class="info-title">Precaution</h3>
                </div>
                <p class="info-text">{data['precaution']}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with info_col2:
            tips_html = "".join([
                f'<li><span class="tip-bullet">→</span>{tip}</li>'
                for tip in data['tips']
            ])
            
            st.markdown(f"""
            <div class="info-card">
                <div class="info-header">
                    <span class="info-icon">💡</span>
                    <h3 class="info-title">Prevention Tips</h3>
                </div>
                <ul class="tips-list">
                    {tips_html}
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    elif confidence >= 40 and disease_name not in disease_data:
        st.markdown("""
        <div class="warning-card" style="margin-top: 32px;">
            <div class="warning-icon">📋</div>
            <p class="warning-title">Information Unavailable</p>
            <p class="warning-text">
                Detailed precautions for this condition are not yet in our database. 
                Please consult a local agricultural expert.
            </p>
        </div>
        """, unsafe_allow_html=True)

# =========================================
# FOOTER
# =========================================

st.markdown("""
<div class="footer">
    <hr>
    <p>Built with 🌱 for healthier crops</p>
    <div class="footer-tech">
        <span class="tech-badge">Streamlit</span>
        <span class="tech-badge">TensorFlow</span>
        <span class="tech-badge">CNN</span>
        <span class="tech-badge">OpenCV</span>
    </div>
</div>
""", unsafe_allow_html=True)