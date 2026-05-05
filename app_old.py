import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import os
from models.numeric_model import NumericModel
from models.image_model import ImageModel
from models.fusion import FusionEngine

# Page Config
st.set_page_config(page_title="Deal Finder AI", page_icon="🏠", layout="wide")

# Advanced Dark Theme UI
st.markdown("""
    <style>
    /* Global Styles */
    .main {
        background: linear-gradient(135deg, #0f0f0f 0%, #1a1a1a 50%, #2d2d2d 100%);
        color: #ffffff;
    }
    
    /* Sidebar Styles */
    .css-1d391kg {
        background: #1a1a1a;
        border-right: 2px solid #333;
    }
    
    /* Input Components */
    .stNumberInput, .stSlider, .stFileUploader {
        background: #2d2d2d;
        border: 2px solid #444;
        color: #ffffff;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.3);
        margin-bottom: 20px;
    }
    
    /* Button Styles */
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        height: 3.5em;
        background: linear-gradient(90deg, #ff6b6b 0%, #ff8e53 100%);
        color: #ffffff;
        font-weight: bold;
        border: none;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(255, 107, 107, 0.3);
    }
    
    .stButton>button:hover {
        transform: translateY(-2px) scale(1.02);
        box-shadow: 0 6px 20px rgba(255, 107, 107, 0.4);
        background: linear-gradient(90deg, #ff8e53 0%, #ff6b6b 100%);
    }
    
    /* Metric Cards */
    .metric-card {
        background: linear-gradient(135deg, #2d2d2d 0%, #1a1a1a 100%);
        padding: 24px;
        border-radius: 15px;
        border-left: 5px solid #ff6b6b;
        box-shadow: 0 10px 30px rgba(0,0,0,0.5);
        color: #ffffff;
        border: 1px solid #444;
    }
    
    /* Deal Badge */
    .deal-badge {
        padding: 12px 24px;
        border-radius: 25px;
        font-weight: bold;
        font-size: 1.3rem;
        text-align: center;
        margin-bottom: 20px;
        box-shadow: 0 5px 15px rgba(0,0,0,0.3);
    }
    
    /* Tab Styles */
    .stTabs [data-baseweb="tab-list"] {
        background: #1a1a1a;
        border-radius: 10px;
        padding: 5px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: #2d2d2d;
        color: #ffffff;
        border-radius: 8px;
        padding: 10px 20px;
        margin: 0 5px;
        transition: all 0.3s ease;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(90deg, #ff6b6b 0%, #ff8e53 100%);
        color: #ffffff;
    }
    
    /* Headers */
    h1, h2, h3, h4, h5, h6 {
        color: #ffffff;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
    }
    
    /* Text */
    .stMarkdown {
        color: #ffffff;
    }
    
    /* Progress Bar */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #ff6b6b 0%, #ff8e53 100%);
    }
    
    /* Info Box */
    .stInfo {
        background: linear-gradient(135deg, #2d2d2d 0%, #1a1a1a 100%);
        border-left: 4px solid #ff6b6b;
        color: #ffffff;
    }
    
    /* Success Box */
    .stSuccess {
        background: linear-gradient(135deg, #1a4d2e 0%, #0d2818 100%);
        border-left: 4px solid #4caf50;
        color: #ffffff;
    }
    
    /* Warning Box */
    .stWarning {
        background: linear-gradient(135deg, #4a3426 0%, #2d1f15 100%);
        border-left: 4px solid #ff9800;
        color: #ffffff;
    }
    
    /* Error Box */
    .stError {
        background: linear-gradient(135deg, #4a1f1f 0%, #2d0f0f 100%);
        border-left: 4px solid #f44336;
        color: #ffffff;
    }
    
    /* Dataframe */
    .stDataFrame {
        background: #2d2d2d;
        color: #ffffff;
        border-radius: 10px;
    }
    
    /* Chart Container */
    .element-container {
        background: rgba(45, 45, 45, 0.3);
        border-radius: 15px;
        padding: 20px;
        margin: 10px 0;
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize Models
@st.cache_resource
def load_models():
    nm = NumericModel()
    # Check if model exists, if not train it
    if not os.path.exists('models/numeric_rf.joblib'):
        nm.train()
    
    im = ImageModel()
    # im.build_model() is no longer needed as __init__ handles it
    
    fe = FusionEngine()
    return nm, im, fe

nm, im, fe = load_models()

# Sidebar Inputs
st.sidebar.header("📍 Property Details")
price = st.sidebar.number_input("Listing Price ($)", min_value=10000, max_value=2000000, value=250000, step=5000)
size = st.sidebar.slider("Size (sqm)", 20, 500, 120)
rooms = st.sidebar.slider("Number of Rooms", 1, 10, 3)

st.sidebar.markdown("---")
uploaded_file = st.sidebar.file_uploader("📸 Upload Property Image", type=["jpg", "jpeg", "png"])

# Main Content
st.title("🏠 Deal Finder AI")
st.markdown("### Evaluate property deals using Data & Vision")

if uploaded_file is not None:
    col1, col2 = st.columns([1, 1])
    
    with col1:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Property Image", use_container_width=True)
        
        # Save temp image for processing
        temp_path = "temp_property_img.png"
        image.save(temp_path)
    
    with col2:
        if st.button("🚀 Analyze Real Estate Deal"):
            with st.spinner("🔬 Processing Hybrid AI Engines..."):
                # 1. Numeric Analysis
                fair_price = nm.predict_fair_price(size, rooms)
                
                # 2. Image Analysis
                condition, confidence = im.predict_condition(temp_path)
                
                # 3. Fusion Analysis
                results = fe.calculate_deal_score(price, fair_price, condition)
                
                # Visual Header
                badge_colors = {
                    "Great Deal": "#28a745",
                    "Good Deal": "#17a2b8",
                    "Fair Deal": "#ffc107",
                    "Bad Deal": "#dc3545",
                    "Risky Bargain": "#fd7e14",
                    "Overpriced": "#6c757d",
                    "Poor Value": "#343a40"
                }
                color = badge_colors.get(results['rating'], "#000")
                st.markdown(f"""
                    <div style="background-color: {color}; color: white; padding: 15px; border-radius: 10px; text-align: center; margin-bottom: 20px;">
                        <h2 style="margin:0; color: white;">{results['rating']}</h2>
                        <span style="font-size: 0.9rem;">Analysis Confidence: {results['score']*100:.1f}%</span>
                    </div>
                """, unsafe_allow_html=True)
                
                # Metrics Section
                st.markdown("### 📊 Key Indicators")
                m1, m2, m3 = st.columns(3)
                with m1:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.metric("Market Valuation", f"${fair_price:,.0f}")
                    st.markdown('</div>', unsafe_allow_html=True)
                with m2:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.metric("Visual Health", condition.capitalize())
                    st.markdown('</div>', unsafe_allow_html=True)
                with m3:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.metric("Price Variance", f"{((price/fair_price-1)*100):.1f}%")
                    st.markdown('</div>', unsafe_allow_html=True)
                
                st.markdown("---")
                st.markdown("### 💡 AI Insights & Reasoning")
                for exp in results['explanations']:
                    st.info(exp)
                
                # Visualization
                import plotly.graph_objects as go
                st.markdown("### 📈 Market Position")
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=['Your Listing', 'Fair Market'],
                    y=[price, fair_price],
                    marker_color=[color, '#1e3c72'],
                    text=[f"${price:,.0f}", f"${fair_price:,.0f}"],
                    textposition='auto',
                ))
                fig.update_layout(height=400, margin=dict(t=20, b=20, l=20, r=20))
                st.plotly_chart(fig, use_container_width=True)
                
            # Cleanup
            if os.path.exists(temp_path):
                os.remove(temp_path)
else:
    st.info("👈 Please enter property details and upload an image in the sidebar to start analysis.")
    
    st.markdown("""
    ### How it works:
    1. **Numerical Analysis**: We use a **Random Forest Regressor** to estimate the property's fair market value based on its size and room count.
    2. **Visual Analysis**: A **MobileNetV2 CNN** classifies the property condition into *Luxury*, *Normal*, or *Damaged*.
    3. **Hybrid Fusion**: Our fusion engine combines price data and visual cues to give you a final 'Deal Score'.
    """)
    
    # Show dummy templates for users to try
    st.markdown("---")
    st.markdown("#### 📥 Don't have an image? Try these templates:")
    t1, t2, t3 = st.columns(3)
    for i, label in enumerate(['luxury', 'normal', 'damaged']):
        path = f"data/images/template_{label}.png"
        if os.path.exists(path):
            with [t1, t2, t3][i]:
                st.image(path, caption=f"Sample {label.capitalize()}", width=150)
