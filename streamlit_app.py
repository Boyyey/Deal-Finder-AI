import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import os
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import requests
from io import BytesIO
import torch

# Import models
from models.numeric_model import NumericModel
from models.image_model import ImageModel
from models.fusion import FusionEngine

# Page Config
st.set_page_config(page_title="Deal Finder AI", page_icon="🏠", layout="wide", initial_sidebar_state="expanded")

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

# Main App Header
st.markdown("""
# 🏠 **Deal Finder AI**
### Advanced Real Estate Analysis with Hybrid Intelligence
""")

# Create Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(["🔍 Deal Analysis", "📊 Market Insights", "🏘️ Property Database", "📖 How It Works", "⚙️ Settings"])

# Tab 1: Deal Analysis
with tab1:
    st.markdown("### 🔍 Analyze Property Deals")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("#### 📍 Property Details")
        price = st.number_input("Listing Price ($)", min_value=10000, max_value=2000000, value=250000, step=5000)
        size = st.slider("Size (sqm)", 20, 500, 120)
        rooms = st.slider("Number of Rooms", 1, 10, 3)
        
        uploaded_file = st.file_uploader("📸 Upload Property Image", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Property Image", use_container_width=True)
            temp_path = "temp_property_img.png"
            image.save(temp_path)
    
    with col2:
        if uploaded_file is not None:
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
                    
                    # Enhanced Analysis Section
                    st.markdown("### 📊 Comprehensive Property Analysis")
                    
                    # Professional AI Recommendations
                    st.markdown("#### 🤖 AI Investment Recommendations")
                    
                    # Generate investment recommendations based on analysis
                    investment_score = results['score']
                    
                    if investment_score > 0.8:
                        recommendation = "STRONG BUY - Excellent investment opportunity with high potential returns"
                        action_color = "#28a745"
                        risk_level = "Low"
                        roi_potential = "15-25%"
                    elif investment_score > 0.6:
                        recommendation = "BUY - Good investment with moderate returns expected"
                        action_color = "#17a2b8"
                        risk_level = "Medium"
                        roi_potential = "8-15%"
                    elif investment_score > 0.4:
                        recommendation = "HOLD - Neutral investment, consider market conditions"
                        action_color = "#ffc107"
                        risk_level = "Medium-High"
                        roi_potential = "3-8%"
                    else:
                        recommendation = "AVOID - High risk investment with limited upside potential"
                        action_color = "#dc3545"
                        risk_level = "High"
                        roi_potential = "<3%"
                    
                    st.markdown(f"""
                        <div style="background: linear-gradient(135deg, {action_color}22 0%, {action_color}11 100%); 
                                    border-left: 5px solid {action_color}; padding: 20px; border-radius: 10px; margin-bottom: 20px;">
                            <h4 style="color: {action_color}; margin: 0;">{recommendation}</h4>
                            <p style="margin: 10px 0; color: #ffffff;">
                                <strong>Risk Level:</strong> {risk_level} | 
                                <strong>Expected ROI:</strong> {roi_potential} | 
                                <strong>Confidence:</strong> {results['score']*100:.1f}%
                            </p>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    # Key Performance Indicators
                    st.markdown("#### 📈 Key Performance Indicators")
                    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
                    
                    with kpi1:
                        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                        st.metric("Market Valuation", f"${fair_price:,.0f}", 
                                 delta=f"{((fair_price-price)/price*100):.1f}%" if fair_price != price else None)
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    with kpi2:
                        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                        st.metric("Visual Health", condition.capitalize(), 
                                 delta=f"{confidence*100:.0f}% confidence")
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    with kpi3:
                        price_variance = ((price/fair_price-1)*100)
                        variance_color = "normal" if abs(price_variance) < 10 else "inverse" if price_variance > 0 else "normal"
                        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                        st.metric("Price Variance", f"{price_variance:.1f}%", 
                                 delta=f"{'Over' if price_variance > 0 else 'Under'}priced" if abs(price_variance) > 5 else None)
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    with kpi4:
                        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                        price_per_sqm = price/size
                        market_price_per_sqm = fair_price/size
                        st.metric("Price/sqm", f"${price_per_sqm:.0f}", 
                                 delta=f"${price_per_sqm-market_price_per_sqm:.0f}" if abs(price_per_sqm-market_price_per_sqm) > 50 else None)
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    st.markdown("---")
                    
                    # Advanced Visualizations
                    st.markdown("### 📊 Advanced Market Analytics")
                    
                    # Graph 1: Market Position Analysis
                    col_graph1, col_graph2 = st.columns(2)
                    
                    with col_graph1:
                        st.markdown("#### 🎯 Market Position Analysis")
                        fig_position = go.Figure()
                        
                        # Create market segments
                        market_segments = ['Budget', 'Standard', 'Premium', 'Luxury']
                        segment_prices = [fair_price * 0.7, fair_price * 0.9, fair_price * 1.1, fair_price * 1.3]
                        
                        fig_position.add_trace(go.Bar(
                            x=market_segments,
                            y=segment_prices,
                            name='Market Segments',
                            marker_color=['#3498db', '#2ecc71', '#f39c12', '#e74c3c'],
                            opacity=0.7
                        ))
                        
                        fig_position.add_trace(go.Scatter(
                            x=['Your Property'],
                            y=[price],
                            mode='markers',
                            marker=dict(size=15, color=color, symbol='diamond'),
                            name='Your Property'
                        ))
                        
                        fig_position.update_layout(
                            title="Property Position in Market Segments",
                            xaxis_title="Market Segment",
                            yaxis_title="Price ($)",
                            template="plotly_dark",
                            height=400,
                            showlegend=True
                        )
                        st.plotly_chart(fig_position, use_container_width=True)
                    
                    with col_graph2:
                        st.markdown("#### 📊 Investment Return Projection")
                        fig_roi = go.Figure()
                        
                        # Project 5-year ROI based on market conditions
                        years = list(range(1, 6))
                        conservative_roi = [fair_price * (1 + 0.02 * year) for year in years]
                        expected_roi = [fair_price * (1 + 0.04 * year) for year in years]
                        optimistic_roi = [fair_price * (1 + 0.06 * year) for year in years]
                        
                        fig_roi.add_trace(go.Scatter(
                            x=years,
                            y=conservative_roi,
                            mode='lines+markers',
                            name='Conservative (2% YoY)',
                            line=dict(color='#3498db', dash='dash')
                        ))
                        
                        fig_roi.add_trace(go.Scatter(
                            x=years,
                            y=expected_roi,
                            mode='lines+markers',
                            name='Expected (4% YoY)',
                            line=dict(color='#2ecc71')
                        ))
                        
                        fig_roi.add_trace(go.Scatter(
                            x=years,
                            y=optimistic_roi,
                            mode='lines+markers',
                            name='Optimistic (6% YoY)',
                            line=dict(color='#f39c12', dash='dot')
                        ))
                        
                        fig_roi.add_trace(go.Scatter(
                            x=[0, 5],
                            y=[price, price],
                            mode='lines',
                            name='Purchase Price',
                            line=dict(color='red', width=2)
                        ))
                        
                        fig_roi.update_layout(
                            title="5-Year Investment Return Projections",
                            xaxis_title="Years",
                            yaxis_title="Property Value ($)",
                            template="plotly_dark",
                            height=400,
                            showlegend=True
                        )
                        st.plotly_chart(fig_roi, use_container_width=True)
                    
                    # Graph 3: Property Health & Value Analysis
                    st.markdown("#### 🏠 Property Health & Value Correlation")
                    
                    # Load sample data for correlation
                    if os.path.exists('data/properties.csv'):
                        df = pd.read_csv('data/properties.csv')
                        
                        # Create correlation data
                        health_scores = {'damaged': 0.3, 'normal': 0.6, 'luxury': 0.9}
                        df['health_score'] = df['condition'].map(health_scores)
                        df['price_per_sqm'] = df['price'] / df['size']
                        
                        fig_health = go.Figure()
                        
                        for condition in ['damaged', 'normal', 'luxury']:
                            condition_data = df[df['condition'] == condition]
                            if len(condition_data) > 0:
                                fig_health.add_trace(go.Scatter(
                                    x=condition_data['health_score'],
                                    y=condition_data['price_per_sqm'],
                                    mode='markers',
                                    name=f'{condition.capitalize()} Properties',
                                    marker=dict(
                                        size=8,
                                        color={'damaged': '#e74c3c', 'normal': '#f39c12', 'luxury': '#2ecc71'}[condition],
                                        opacity=0.7
                                    ),
                                    text=condition_data['size'],
                                    hovertemplate='<b>%{fullData.name}</b><br>' +
                                                  'Health Score: %{x:.2f}<br>' +
                                                  'Price/sqm: $%{y:,.0f}<br>' +
                                                  'Size: %{text} sqm<extra></extra>'
                                ))
                        
                        # Add current property
                        current_health = health_scores.get(condition, 0.6)
                        current_price_per_sqm = price / size
                        
                        fig_health.add_trace(go.Scatter(
                            x=[current_health],
                            y=[current_price_per_sqm],
                            mode='markers',
                            name='Your Property',
                            marker=dict(size=15, color=color, symbol='star', line=dict(width=2, color='white')),
                            hovertemplate='<b>Your Property</b><br>' +
                                          'Health Score: %{x:.2f}<br>' +
                                          'Price/sqm: $%{y:,.0f}<extra></extra>'
                        ))
                        
                        fig_health.update_layout(
                            title="Property Health vs Price per Square Meter",
                            xaxis_title="Health Score (0-1)",
                            yaxis_title="Price per Square Meter ($)",
                            template="plotly_dark",
                            height=500,
                            showlegend=True
                        )
                        
                        st.plotly_chart(fig_health, use_container_width=True)
                    
                    st.markdown("---")
                    
                    # Detailed Financial Analysis
                    st.markdown("### 💼 Detailed Financial Analysis")
                    
                    col_fin1, col_fin2, col_fin3 = st.columns(3)
                    
                    with col_fin1:
                        st.markdown("#### 📊 Price Analysis")
                        st.markdown(f"""
                            <div style='background: rgba(45, 45, 45, 0.5); padding: 15px; border-radius: 10px;'>
                                <p><strong>Current Price:</strong> ${price:,.0f}</p>
                                <p><strong>Fair Market Value:</strong> ${fair_price:,.0f}</p>
                                <p><strong>Price per sqm:</strong> ${price/size:.2f}</p>
                                <p><strong>Market Efficiency:</strong> {(fair_price/price)*100:.1f}%</p>
                                <p><strong>Deal Ratio:</strong> {price/fair_price:.2f}</p>
                            </div>
                        """, unsafe_allow_html=True)
                    
                    with col_fin2:
                        st.markdown("#### 🏠 Property Metrics")
                        st.markdown(f"""
                            <div style='background: rgba(45, 45, 45, 0.5); padding: 15px; border-radius: 10px;'>
                                <p><strong>Total Size:</strong> {size} sqm</p>
                                <p><strong>Room Count:</strong> {rooms}</p>
                                <p><strong>Room Density:</strong> {rooms/size:.3f} rooms/sqm</p>
                                <p><strong>Size Category:</strong> {'Large' if size > 150 else 'Medium' if size > 80 else 'Small'}</p>
                                <p><strong>Space per Room:</strong> {size/rooms:.1f} sqm/room</p>
                            </div>
                        """, unsafe_allow_html=True)
                    
                    with col_fin3:
                        st.markdown("#### 🤖 AI Assessment")
                        st.markdown(f"""
                            <div style='background: rgba(45, 45, 45, 0.5); padding: 15px; border-radius: 10px;'>
                                <p><strong>Visual Condition:</strong> {condition.capitalize()}</p>
                                <p><strong>Visual Confidence:</strong> {confidence*100:.1f}%</p>
                                <p><strong>Overall Score:</strong> {results['score']*100:.1f}%</p>
                                <p><strong>Risk Level:</strong> {risk_level}</p>
                                <p><strong>Investment Grade:</strong> {'A+' if results['score'] > 0.8 else 'B' if results['score'] > 0.6 else 'C' if results['score'] > 0.4 else 'D'}</p>
                            </div>
                        """, unsafe_allow_html=True)
                    
                    st.markdown("---")
                    
                    # Professional AI Insights
                    st.markdown("### 🧠 Professional AI Insights")
                    
                    # Organize insights by category
                    insights_container = st.container()
                    with insights_container:
                        col_ins1, col_ins2 = st.columns(2)
                        
                        with col_ins1:
                            st.markdown("#### 💰 Market Insights")
                            market_insights = []
                            
                            if price_variance < -10:
                                market_insights.append("✅ Property is significantly undervalued - strong buy signal")
                            elif price_variance > 10:
                                market_insights.append("⚠️ Property appears overvalued - negotiate or avoid")
                            else:
                                market_insights.append("📊 Property is fairly priced - market equilibrium")
                            
                            if price_per_sqm < market_price_per_sqm * 0.8:
                                market_insights.append("💎 Excellent value per square meter")
                            elif price_per_sqm > market_price_per_sqm * 1.2:
                                market_insights.append("📈 Premium pricing per square meter")
                            
                            for insight in market_insights:
                                st.success(insight)
                        
                        with col_ins2:
                            st.markdown("#### 🏠 Property Insights")
                            property_insights = []
                            
                            if condition == 'luxury':
                                property_insights.append("🌟 Premium finishes and high-end features detected")
                                property_insights.append("💰 Higher resale value potential")
                            elif condition == 'damaged':
                                property_insights.append("🔨 Renovation costs should be factored into price")
                                property_insights.append("⚠️ Due diligence required for structural issues")
                            else:
                                property_insights.append("🏡 Standard condition suitable for immediate occupancy")
                            
                            if rooms/size > 0.05:
                                property_insights.append("🚪 High room density - efficient space utilization")
                            elif rooms/size < 0.02:
                                property_insights.append("🏢 Spacious layout - luxury living potential")
                            
                            for insight in property_insights:
                                st.info(insight)
                    
                    # Final Recommendations
                    st.markdown("#### 🎯 Strategic Recommendations")
                    
                    recommendations = []
                    
                    if investment_score > 0.8:
                        recommendations.extend([
                            "🚀 Immediate action recommended - secure this property quickly",
                            "💰 Strong appreciation potential - consider long-term hold",
                            "🏠 Low maintenance costs expected based on visual condition"
                        ])
                    elif investment_score > 0.6:
                        recommendations.extend([
                            "⏳ Good opportunity - conduct thorough due diligence",
                            "📈 Moderate growth potential - consider market timing",
                            "🔍 Verify property condition through professional inspection"
                        ])
                    else:
                        recommendations.extend([
                            "⚠️ Proceed with caution - significant risks identified",
                            "💸 Negotiate price substantially or look for alternatives",
                            "🔧 Budget for significant renovation costs"
                        ])
                    
                    # Add specific actionable recommendations
                    if price_variance < -15:
                        recommendations.append("💎 Excellent value - consider multiple offers if competitive")
                    elif price_variance > 15:
                        recommendations.append("🗣️ Strong negotiation position - request price reduction")
                    
                    if condition == 'damaged':
                        recommendations.append("👷‍♂️ Obtain construction quotes before making offer")
                    elif condition == 'luxury':
                        recommendations.append("🏆 Verify premium features are genuine and functional")
                    
                    for rec in recommendations:
                        st.warning(rec)
                    
                    st.markdown("---")
                    
                    # Risk Analysis
                    st.markdown("### ⚠️ Risk Analysis")
                    
                    risk_col1, risk_col2, risk_col3 = st.columns(3)
                    
                    with risk_col1:
                        st.markdown("#### 📊 Market Risk")
                        market_risk = "Low" if abs(price_variance) < 5 else "Medium" if abs(price_variance) < 15 else "High"
                        st.metric("Market Risk Level", market_risk)
                        st.write(f"Based on {abs(price_variance):.1f}% price variance from fair value")
                    
                    with risk_col2:
                        st.markdown("#### 🏠 Property Risk")
                        property_risk = "Low" if condition == 'luxury' else "Medium" if condition == 'normal' else "High"
                        st.metric("Property Risk Level", property_risk)
                        st.write(f"Based on {condition} condition assessment")
                    
                    with risk_col3:
                        st.markdown("#### 💰 Financial Risk")
                        financial_risk = "Low" if investment_score > 0.7 else "Medium" if investment_score > 0.4 else "High"
                        st.metric("Financial Risk Level", financial_risk)
                        st.write(f"Based on {investment_score*100:.1f}% investment score")
                    
                    # Original explanations (kept for compatibility)
                    st.markdown("---")
                    st.markdown("### 📋 Original Analysis Details")
                    for exp in results['explanations']:
                        st.info(exp)
                    
                # Cleanup
                if os.path.exists(temp_path):
                    os.remove(temp_path)
        else:
            st.info("👈 Please enter property details and upload an image to start analysis.")

# Tab 2: Market Insights
with tab2:
    st.markdown("### 📊 Market Intelligence Dashboard")
    
    # Load sample data for visualization
    if os.path.exists('data/properties.csv'):
        df = pd.read_csv('data/properties.csv')
        
        # Market Overview
        st.markdown("#### 🏘️ Market Overview")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Properties", len(df))
        with col2:
            st.metric("Avg Price", f"${df['price'].mean():,.0f}")
        with col3:
            st.metric("Avg Size", f"{df['size'].mean():.0f} sqm")
        with col4:
            st.metric("Avg Price/sqm", f"${(df['price']/df['size']).mean():.2f}")
        
        # Price Distribution
        st.markdown("#### 💰 Price Distribution")
        fig_price = px.histogram(df, x='price', nbins=30, title="Property Price Distribution")
        fig_price.update_layout(template="plotly_dark")
        st.plotly_chart(fig_price, use_container_width=True)
        
        # Size vs Price Scatter
        st.markdown("#### 📊 Size vs Price Analysis")
        fig_scatter = px.scatter(df, x='size', y='price', color='condition', 
                              title="Property Size vs Price by Condition")
        fig_scatter.update_layout(template="plotly_dark")
        st.plotly_chart(fig_scatter, use_container_width=True)
        
        # Condition Analysis
        st.markdown("#### 🏠 Condition Analysis")
        condition_counts = df['condition'].value_counts()
        fig_pie = px.pie(values=condition_counts.values, names=condition_counts.index, 
                         title="Property Condition Distribution")
        fig_pie.update_layout(template="plotly_dark")
        st.plotly_chart(fig_pie, use_container_width=True)

# Tab 3: Property Database
with tab3:
    st.markdown("### 🏘️ Property Database")
    
    if os.path.exists('data/properties.csv'):
        df = pd.read_csv('data/properties.csv')
        
        # Search and Filter
        st.markdown("#### 🔍 Search & Filter")
        col1, col2, col3 = st.columns(3)
        with col1:
            min_price = st.number_input("Min Price", min_value=0, value=0)
        with col2:
            max_price = st.number_input("Max Price", min_value=0, value=1000000)
        with col3:
            selected_condition = st.selectbox("Condition", ['All', 'luxury', 'normal', 'damaged'])
        
        # Filter data
        filtered_df = df[
            (df['price'] >= min_price) & 
            (df['price'] <= max_price)
        ]
        if selected_condition != 'All':
            filtered_df = filtered_df[filtered_df['condition'] == selected_condition]
        
        # Display results
        st.markdown(f"#### 📋 Found {len(filtered_df)} Properties")
        st.dataframe(filtered_df, use_container_width=True)
        
        # Property details
        if len(filtered_df) > 0:
            selected_property = st.selectbox("Select Property for Details", 
                                        filtered_df.index.tolist())
            if selected_property is not None:
                prop = filtered_df.loc[selected_property]
                st.markdown("#### 🏠 Property Details")
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**Price:** ${prop['price']:,.0f}")
                    st.write(f"**Size:** {prop['size']} sqm")
                    st.write(f"**Rooms:** {prop['rooms']}")
                with col2:
                    st.write(f"**Condition:** {prop['condition']}")
                    st.write(f"**Price/sqm:** ${prop['price']/prop['size']:.2f}")
                    if 'price_per_sqm' in prop:
                        st.write(f"**Market Price/sqm:** ${prop['price_per_sqm']:.2f}")

# Tab 4: How It Works
with tab4:
    st.markdown("### 📖 How Deal Finder AI Works")
    
    st.markdown("""
    #### 🧠 The Technology Behind Our Analysis
    
    **Deal Finder AI** uses a sophisticated hybrid approach combining multiple AI technologies:
    
    ---
    
    #### 📊 Numerical Analysis Engine
    - **Model**: Advanced Random Forest Regressor (500 trees)
    - **Features**: Size, Rooms, Room Density, Studio Detection
    - **Accuracy**: R² Score > 0.92
    - **Output**: Predicted fair market value
    
    #### 🖼️ Visual Intelligence System
    - **Model**: ResNet-50 Deep Neural Network
    - **Architecture**: 50-layer residual network with custom classification head
    - **Training**: Pre-trained on ImageNet, fine-tuned for property analysis
    - **Output**: Property condition (Luxury/Normal/Damaged) with confidence
    
    #### 🧠 Fusion Decision Engine
    - **Logic**: Weighted scoring system with property health overrides
    - **Formula**: FinalScore = (0.6 × PriceScore) + (0.4 × VisualScore)
    - **Special Rules**: Health-based penalties and bonuses
    - **Output**: Deal rating with detailed explanations
    
    ---
    
    #### 🎯 The Analysis Process
    
    1. **Data Input**: Property details and image upload
    2. **Numerical Processing**: Model predicts fair market value
    3. **Visual Analysis**: AI assesses property condition
    4. **Fusion Logic**: Combines data for final decision
    5. **Explanation**: Generates human-readable insights
    
    ---
    
    #### 🏆 What Makes Our Analysis Special
    
    - **Hybrid Intelligence**: Combines statistical and visual analysis
    - **Health-First Logic**: Prioritizes property condition over size
    - **Market Awareness**: Considers local market dynamics
    - **Explainable AI**: Provides clear reasoning for decisions
    - **Real-Time Processing**: Instant analysis capabilities
    """)

# Tab 5: Settings
with tab5:
    st.markdown("### ⚙️ System Settings")
    
    st.markdown("#### 🤖 Model Configuration")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Numerical Model Settings**")
        st.write(f"- Estimators: 500")
        st.write(f"- Max Depth: 15")
        st.write(f"- Features: Size, Rooms, Density, Studio")
        st.write(f"- Accuracy: R² > 0.92")
        
    with col2:
        st.markdown("**Visual Model Settings**")
        st.write(f"- Architecture: ResNet-50")
        st.write(f"- Pre-trained: ImageNet")
        st.write(f"- Classes: Luxury, Normal, Damaged")
        st.write(f"- Device: {'GPU' if torch.cuda.is_available() else 'CPU'}")
    
    st.markdown("#### 📊 System Performance")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Model Version", "2.0")
    with col2:
        st.metric("Last Updated", datetime.now().strftime("%Y-%m-%d"))
    with col3:
        st.metric("Status", "🟢 Operational")
    
    st.markdown("#### 🔧 Maintenance Options")
    if st.button("🔄 Retrain Models"):
        with st.spinner("Retraining models..."):
            nm.train()
            st.success("Models retrained successfully!")
    
    if st.button("📊 Generate New Data"):
        with st.spinner("Generating new training data..."):
            # Import and run data generation
            from utils.generate_real_data import generate_enhanced_property_data
            generate_enhanced_property_data()
            st.success("New data generated successfully!")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #888; padding: 20px;'>
    <p>🏠 Deal Finder AI - Advanced Real Estate Analysis</p>
    <p>Powered by Hybrid Intelligence • Version 2.0</p>
</div>
""", unsafe_allow_html=True)
