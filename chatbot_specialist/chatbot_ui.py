import streamlit as st
import streamlit.components.v1 as components
import requests
import json
import logging
import base64
from pathlib import Path
from datetime import datetime
import pandas as pd
import uuid
from typing import Optional, Dict, Any, List


# --- FUNZIONE PER CODIFICARE L'IMMAGINE IN BASE64 ---
def img_to_base64(image_path):
    """Codifica un'immagine in Base64 per l'uso in HTML."""
    path = Path(image_path)
    if not path.is_file():
        return None
    with path.open("rb") as img_file:
        return base64.b64encode(img_file.read()).decode()


# Configurazione logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Configurazione dell'App Streamlit ---
st.set_page_config(
    page_title="BlueAI - Blues System", layout="wide", initial_sidebar_state="expanded"
)

# --- CSS PERSONALIZZATO ULTRA FUTURISTICO ---
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600;700&display=swap');
    
    /* Reset */
    * {
        font-family: 'Inter', sans-serif;
    }
    
    /* Sfondo animato con particelle */
    .stApp {
        background: linear-gradient(-45deg, #0a0e27, #1a1f4d, #2a3f7f, #1e40af);
        background-size: 400% 400%;
        animation: gradientFlow 20s ease infinite;
        min-height: 100vh;
        position: relative;
        overflow-x: hidden;
    }
    
    @keyframes gradientFlow {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    /* Effetto particelle di sfondo */
    .stApp::before {
        content: '';
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background-image: 
            radial-gradient(2px 2px at 20% 30%, rgba(59, 130, 246, 0.4), transparent),
            radial-gradient(2px 2px at 60% 70%, rgba(147, 197, 253, 0.3), transparent),
            radial-gradient(1px 1px at 50% 50%, rgba(191, 219, 254, 0.2), transparent),
            radial-gradient(1px 1px at 80% 10%, rgba(59, 130, 246, 0.3), transparent);
        background-size: 200% 200%;
        animation: particleFloat 15s ease infinite;
        pointer-events: none;
        z-index: 0;
    }
    
    @keyframes particleFloat {
        0%, 100% { transform: translate(0, 0); }
        33% { transform: translate(30px, -30px); }
        66% { transform: translate(-20px, 20px); }
    }
    
    /* Container principale con effetto vetro */
    .main .block-container {
        background: rgba(15, 23, 42, 0.75);
        border-radius: 28px;
        padding: 2.5rem;
        margin-top: 1rem;
        box-shadow: 
            0 25px 60px rgba(0, 0, 0, 0.5),
            inset 0 1px 0 rgba(255, 255, 255, 0.1),
            0 0 100px rgba(59, 130, 246, 0.1);
        backdrop-filter: blur(30px) saturate(180%);
        border: 1px solid rgba(59, 130, 246, 0.25);
        position: relative;
        z-index: 1;
    }
    
    /* Header epico con mega glow */
    .custom-header {
        text-align: center;
        padding: 2rem 0 1rem 0;
        color: #ffffff;
        font-size: 4rem;
        font-weight: 900;
        margin-bottom: 0.5rem;
        letter-spacing: -3px;
        text-shadow: 
            0 0 20px rgba(59, 130, 246, 1),
            0 0 40px rgba(59, 130, 246, 0.8),
            0 0 60px rgba(59, 130, 246, 0.6),
            0 0 80px rgba(59, 130, 246, 0.4),
            4px 4px 8px rgba(0, 0, 0, 0.8);
        animation: megaGlow 3s ease-in-out infinite alternate;
        position: relative;
    }
    
    @keyframes megaGlow {
        0% { 
            text-shadow: 
                0 0 20px rgba(59, 130, 246, 1),
                0 0 40px rgba(59, 130, 246, 0.8),
                0 0 60px rgba(59, 130, 246, 0.6);
        }
        100% { 
            text-shadow: 
                0 0 30px rgba(59, 130, 246, 1),
                0 0 60px rgba(59, 130, 246, 1),
                0 0 90px rgba(59, 130, 246, 0.8),
                0 0 120px rgba(59, 130, 246, 0.6);
        }
    }
    
    .custom-subtitle {
        text-align: center;
        color: #cbd5e1;
        font-size: 1.3rem;
        margin-bottom: 2rem;
        font-weight: 600;
        text-shadow: 
            0 0 10px rgba(59, 130, 246, 0.5),
            2px 2px 4px rgba(0, 0, 0, 0.5);
        letter-spacing: 1px;
        animation: subtitlePulse 4s ease-in-out infinite;
    }
    
    @keyframes subtitlePulse {
        0%, 100% { opacity: 0.9; }
        50% { opacity: 1; }
    }
    
    .main h1 { display: none; }
    
    /* Chat Messages ultra smooth */
    .stChatMessage {
        margin-bottom: 1.8rem;
        border-radius: 24px;
        border: none;
        animation: messageSlide 0.6s cubic-bezier(0.34, 1.56, 0.64, 1);
        transition: all 0.4s cubic-bezier(0.34, 1.56, 0.64, 1);
    }
    
    @keyframes messageSlide {
        from {
            opacity: 0;
            transform: translateY(40px) scale(0.9) rotateX(10deg);
        }
        to {
            opacity: 1;
            transform: translateY(0) scale(1) rotateX(0deg);
        }
    }
    
    .stChatMessage:hover {
        transform: translateY(-4px) scale(1.01);
        box-shadow: 0 15px 40px rgba(59, 130, 246, 0.3);
    }
    
    /* Messaggio utente futuristico */
    div[data-testid="chat-message-user"] {
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 50%, #1d4ed8 100%);
        color: white;
        border-radius: 24px 24px 6px 24px;
        padding: 1.4rem 2rem;
        margin-left: 12%;
        box-shadow: 
            0 10px 30px rgba(59, 130, 246, 0.5),
            inset 0 1px 0 rgba(255, 255, 255, 0.25),
            0 0 60px rgba(59, 130, 246, 0.2);
        position: relative;
        overflow: hidden;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    div[data-testid="chat-message-user"]::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: linear-gradient(45deg, transparent, rgba(255, 255, 255, 0.15), transparent);
        transform: rotate(45deg);
        animation: megaShine 4s infinite;
    }
    
    @keyframes megaShine {
        0% { transform: translateX(-100%) translateY(-100%) rotate(45deg); }
        100% { transform: translateX(100%) translateY(100%) rotate(45deg); }
    }
    
    div[data-testid="chat-message-user"] p {
        color: white;
        font-weight: 600;
        font-size: 1.08rem;
        margin: 0;
        position: relative;
        z-index: 1;
        line-height: 1.6;
    }
    
    /* Messaggio assistente high-tech */
    div[data-testid="chat-message-assistant"] {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.95) 0%, rgba(15, 23, 42, 0.98) 100%);
        border-radius: 24px 24px 24px 6px;
        padding: 1.4rem 2rem;
        margin-right: 12%;
        border-left: 5px solid #3b82f6;
        box-shadow: 
            0 10px 30px rgba(0, 0, 0, 0.4),
            inset 0 1px 0 rgba(59, 130, 246, 0.15),
            0 0 40px rgba(59, 130, 246, 0.1);
        position: relative;
        border-top: 1px solid rgba(59, 130, 246, 0.2);
        border-right: 1px solid rgba(59, 130, 246, 0.1);
    }
    
    div[data-testid="chat-message-assistant"]::after {
        content: '';
        position: absolute;
        bottom: 0;
        left: 0;
        width: 100%;
        height: 2px;
        background: linear-gradient(90deg, transparent, #3b82f6, transparent);
        opacity: 0.6;
        animation: borderGlow 3s ease-in-out infinite;
    }
    
    @keyframes borderGlow {
        0%, 100% { opacity: 0.4; }
        50% { opacity: 0.8; }
    }
    
    div[data-testid="chat-message-assistant"] p {
        color: #e2e8f0;
        font-weight: 500;
        font-size: 1.08rem;
        margin: 0;
        line-height: 1.8;
    }
    
    /* Avatar con mega pulse */
    div[data-testid="chat-message-user"] img,
    div[data-testid="chat-message-assistant"] img {
        border-radius: 50%;
        padding: 0.5rem;
        box-shadow: 0 0 30px rgba(59, 130, 246, 0.8);
        animation: avatarPulse 2.5s ease-in-out infinite;
    }
    
    @keyframes avatarPulse {
        0%, 100% { 
            box-shadow: 0 0 20px rgba(59, 130, 246, 0.6);
            transform: scale(1);
        }
        50% { 
            box-shadow: 0 0 40px rgba(59, 130, 246, 1);
            transform: scale(1.05);
        }
    }
    
    div[data-testid="chat-message-user"] img {
        background: linear-gradient(135deg, #1e40af, #3b82f6);
    }
    
    div[data-testid="chat-message-assistant"] img {
        background: linear-gradient(135deg, #3b82f6, #06b6d4);
    }
    
    /* Input area spaziale */
    .stChatInputContainer {
        background: rgba(20, 30, 50, 0.98);
        border-radius: 35px;
        padding: 0.85rem;
        box-shadow: 
            0 20px 50px rgba(0, 0, 0, 0.5),
            inset 0 1px 0 rgba(255, 255, 255, 0.1),
            0 0 80px rgba(59, 130, 246, 0.3);
        border: 2px solid rgba(59, 130, 246, 0.7);
        backdrop-filter: blur(25px);
        transition: all 0.4s cubic-bezier(0.34, 1.56, 0.64, 1);
        position: relative;
    }
    
    .stChatInputContainer::before {
        content: '';
        position: absolute;
        top: -2px;
        left: -2px;
        right: -2px;
        bottom: -2px;
        background: linear-gradient(45deg, #3b82f6, #06b6d4, #3b82f6);
        border-radius: 35px;
        opacity: 0;
        transition: opacity 0.3s;
        z-index: -1;
        animation: borderRotate 3s linear infinite;
    }
    
    @keyframes borderRotate {
        100% { transform: rotate(360deg); }
    }
    
    .stChatInputContainer:focus-within {
        border-color: #3b82f6;
        box-shadow: 
            0 20px 60px rgba(59, 130, 246, 0.6),
            inset 0 1px 0 rgba(255, 255, 255, 0.15),
            0 0 100px rgba(59, 130, 246, 0.4);
        transform: translateY(-2px);
    }
    
    .stChatInputContainer:focus-within::before {
        opacity: 0.5;
    }
    
    .stChatInputContainer textarea {
        background: transparent;
        border: none;
        color: #ffffff;
        font-size: 1.1rem;
        font-weight: 500;
        padding: 1.1rem 1.6rem;
        border-radius: 30px;
        resize: none;
    }
    
    .stChatInputContainer textarea:focus {
        outline: none;
        box-shadow: none;
    }
    
    .stChatInputContainer textarea::placeholder {
        color: #94a3b8;
        font-style: italic;
        font-weight: 400;
    }
    
    /* Pulsante invio MEGA futuristico */
    .stChatInputContainer button {
        background: linear-gradient(135deg, #3b82f6, #1d4ed8);
        color: white;
        border: none;
        border-radius: 50%;
        width: 60px;
        height: 60px;
        box-shadow: 
            0 10px 25px rgba(59, 130, 246, 0.6),
            0 0 40px rgba(59, 130, 246, 0.4);
        transition: all 0.4s cubic-bezier(0.34, 1.56, 0.64, 1);
        position: relative;
        overflow: hidden;
    }
    
    .stChatInputContainer button::before {
        content: '';
        position: absolute;
        top: 50%;
        left: 50%;
        width: 0;
        height: 0;
        border-radius: 50%;
        background: rgba(255, 255, 255, 0.4);
        transform: translate(-50%, -50%);
        transition: width 0.6s, height 0.6s;
    }
    
    .stChatInputContainer button:hover::before {
        width: 400px;
        height: 400px;
    }
    
    .stChatInputContainer button:hover {
        transform: translateY(-4px) rotate(90deg) scale(1.1);
        box-shadow: 
            0 15px 40px rgba(59, 130, 246, 0.8),
            0 0 60px rgba(59, 130, 246, 0.6);
    }
    
    .stChatInputContainer button:active {
        transform: translateY(-2px) rotate(90deg) scale(1.05);
    }
    
    /* Code blocks sci-fi */
    .stCodeBlock {
        background: linear-gradient(135deg, #0a0e1f, #1a1f3a);
        border: 1px solid #334155;
        border-radius: 18px;
        margin: 1.8rem 0;
        overflow: hidden;
        box-shadow: 
            0 10px 30px rgba(0, 0, 0, 0.5),
            inset 0 1px 0 rgba(59, 130, 246, 0.3),
            0 0 40px rgba(59, 130, 246, 0.1);
        position: relative;
    }
    
    .stCodeBlock::before {
        content: '▸ CYPHER QUERY';
        position: absolute;
        top: 10px;
        right: 15px;
        font-size: 0.7rem;
        font-weight: 800;
        color: #3b82f6;
        letter-spacing: 1.5px;
        opacity: 0.8;
        text-shadow: 0 0 10px rgba(59, 130, 246, 0.8);
    }
    
    .stCodeBlock code {
        color: #06b6d4;
        font-family: 'JetBrains Mono', monospace;
        font-size: 1rem;
        line-height: 1.7;
        text-shadow: 0 0 15px rgba(6, 182, 212, 0.4);
        font-weight: 500;
    }
    
    /* Spinner ultra tech */
    .stSpinner > div {
        border-color: #3b82f6 transparent #06b6d4 transparent;
        border-width: 5px;
        animation: spinTech 1.2s cubic-bezier(0.68, -0.55, 0.265, 1.55) infinite;
    }
    
    @keyframes spinTech {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    /* Alerts futuristici */
    .stAlert {
        border-radius: 18px;
        border: none;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);
        backdrop-filter: blur(15px);
        animation: alertSlide 0.5s ease-out;
        border-left: 5px solid;
    }
    
    @keyframes alertSlide {
        from { 
            opacity: 0;
            transform: translateX(-30px);
        }
        to { 
            opacity: 1;
            transform: translateX(0);
        }
    }
    
    .stError {
        background: linear-gradient(135deg, rgba(254, 242, 242, 0.95), rgba(254, 226, 226, 0.95));
        color: #dc2626;
        border-left-color: #dc2626;
    }
    
    .stSuccess {
        background: linear-gradient(135deg, rgba(240, 253, 244, 0.95), rgba(220, 252, 231, 0.95));
        color: #16a34a;
        border-left-color: #16a34a;
    }
    
    .stWarning {
        background: linear-gradient(135deg, rgba(254, 252, 232, 0.95), rgba(254, 249, 195, 0.95));
        color: #ca8a04;
        border-left-color: #ca8a04;
    }
    
    .stInfo {
        background: linear-gradient(135deg, rgba(239, 246, 255, 0.95), rgba(219, 234, 254, 0.95));
        color: #2563eb;
        border-left-color: #2563eb;
    }
    
    /* Sidebar mega moderna */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(10, 15, 30, 0.98), rgba(20, 30, 50, 0.98));
        backdrop-filter: blur(25px);
        border-right: 1px solid rgba(59, 130, 246, 0.3);
        box-shadow: 5px 0 30px rgba(59, 130, 246, 0.1);
    }
    
    [data-testid="stSidebar"] h1, 
    [data-testid="stSidebar"] h2, 
    [data-testid="stSidebar"] h3 {
        color: #ffffff;
        text-shadow: 0 0 15px rgba(59, 130, 246, 0.6);
        font-weight: 700;
    }
    
    [data-testid="stSidebar"] .stButton button {
        width: 100%;
        background: linear-gradient(135deg, rgba(59, 130, 246, 0.25), rgba(29, 78, 216, 0.25));
        color: #ffffff;
        border: 1px solid rgba(59, 130, 246, 0.5);
        border-radius: 14px;
        padding: 0.85rem 1.2rem;
        font-weight: 700;
        font-size: 0.95rem;
        transition: all 0.4s cubic-bezier(0.34, 1.56, 0.64, 1);
        margin: 0.6rem 0;
        box-shadow: 0 4px 15px rgba(59, 130, 246, 0.2);
        position: relative;
        overflow: hidden;
    }
    
    [data-testid="stSidebar"] .stButton button::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.2), transparent);
        transition: left 0.5s;
    }
    
    [data-testid="stSidebar"] .stButton button:hover::before {
        left: 100%;
    }
    
    [data-testid="stSidebar"] .stButton button:hover {
        background: linear-gradient(135deg, rgba(59, 130, 246, 0.45), rgba(29, 78, 216, 0.45));
        transform: translateX(8px) scale(1.02);
        box-shadow: 0 6px 20px rgba(59, 130, 246, 0.4);
        border-color: #3b82f6;
    }
    
    /* Expander futuristico */
    .streamlit-expanderHeader {
        background: rgba(30, 41, 59, 0.6);
        border-radius: 14px;
        border: 1px solid rgba(59, 130, 246, 0.4);
        color: #cbd5e1;
        font-weight: 700;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
    }
    
    .streamlit-expanderHeader:hover {
        background: rgba(30, 41, 59, 0.9);
        border-color: rgba(59, 130, 246, 0.7);
        box-shadow: 0 6px 20px rgba(59, 130, 246, 0.3);
    }
    
    /* Scrollbar cyber */
    ::-webkit-scrollbar {
        width: 12px;
        height: 12px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(15, 23, 42, 0.6);
        border-radius: 6px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(180deg, #3b82f6, #1d4ed8);
        border-radius: 6px;
        box-shadow: 0 0 15px rgba(59, 130, 246, 0.6);
        border: 2px solid rgba(15, 23, 42, 0.6);
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(180deg, #2563eb, #1e40af);
        box-shadow: 0 0 25px rgba(59, 130, 246, 0.9);
    }
    
    /* Badge stile futuristico */
    .status-badge {
        display: inline-block;
        padding: 0.4rem 1rem;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.85rem;
        letter-spacing: 0.5px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
        animation: badgePulse 2s ease-in-out infinite;
    }
    
    @keyframes badgePulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.8; }
    }
    
    .status-online {
        background: linear-gradient(135deg, #10b981, #059669);
        color: white;
        box-shadow: 0 0 20px rgba(16, 185, 129, 0.5);
    }
    
    .status-offline {
        background: linear-gradient(135deg, #ef4444, #dc2626);
        color: white;
        box-shadow: 0 0 20px rgba(239, 68, 68, 0.5);
    }
    
    /* Footer sticky futuristico */
    .footer-sticky {
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        background: rgba(10, 15, 30, 0.95);
        backdrop-filter: blur(20px);
        padding: 1rem 0;
        border-top: 1px solid rgba(59, 130, 246, 0.3);
        box-shadow: 0 -10px 40px rgba(0, 0, 0, 0.5);
        z-index: 999;
        text-align: center;
        color: #94a3b8;
        font-size: 0.9rem;
    }
    
    .footer-sticky strong {
        color: #cbd5e1;
        font-weight: 700;
    }
    
    /* Animazione icona */
    @keyframes floatIcon {
        0%, 100% { transform: translateY(0) rotate(0deg); }
        25% { transform: translateY(-8px) rotate(5deg); }
        75% { transform: translateY(-4px) rotate(-5deg); }
    }
    
    .float-icon {
        animation: floatIcon 4s ease-in-out infinite;
        display: inline-block;
    }
    
    /* Metric cards futuristiche */
    .metric-card {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.8), rgba(15, 23, 42, 0.8));
        border-radius: 16px;
        padding: 1.5rem;
        border: 1px solid rgba(59, 130, 246, 0.3);
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.3);
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 35px rgba(59, 130, 246, 0.3);
        border-color: rgba(59, 130, 246, 0.6);
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: rgba(30, 41, 59, 0.5);
        border-radius: 14px;
        padding: 0.5rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border-radius: 10px;
        color: #94a3b8;
        font-weight: 600;
        padding: 0.75rem 1.5rem;
        transition: all 0.3s ease;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(59, 130, 246, 0.2);
        color: #ffffff;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #3b82f6, #2563eb);
        color: #ffffff;
        box-shadow: 0 4px 15px rgba(59, 130, 246, 0.5);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- HEADER CON LOGO ---
LOGO_PATH = "logo/logo.png"
logo_base64 = img_to_base64(LOGO_PATH)

if logo_base64:
    logo_html = f'<img src="data:image/png;base64,{logo_base64}" alt="Logo" style="width: 80px; vertical-align: middle; margin-right: 20px; filter: drop-shadow(0 0 30px rgba(59, 130, 246, 1));">'
else:
    logo_html = '<span style="font-size: 4rem; filter: drop-shadow(0 0 20px rgba(59, 130, 246, 0.8));">🤖</span>'

st.markdown(
    f"""
    <div class="custom-header">
        {logo_html}BlueAI
    </div>
    <div class="custom-subtitle">
         Sistema AI Avanzato per Knowledge Graph • Neo4j Powered
    </div>
    """,
    unsafe_allow_html=True,
)

query_params = st.query_params
is_debug_mode = query_params.get("debug", "false").lower() == "false"

# 2. Se NON siamo in debug, nascondi la sidebar con il CSS
if not is_debug_mode:
    st.markdown(
        """
        <style>
            [data-testid="stSidebar"] {
                display: none;
            }
            /* Espande il container principale per prendere tutto lo spazio */
            .main .block-container {
                max-width: 100%; 
                padding: 2.5rem 4rem; /* Aggiungi un po' di padding laterale */
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

# URL del backend
API_URL = "http://localhost:8000/ask"
HEALTH_URL = "http://localhost:8000/health"
CACHE_URL = "http://localhost:8000/cache"
CONVERSATION_URL = "http://localhost:8000/conversation"
FEEDBACK_URL = "http://localhost:8000/feedback"
GRAPH_EXPAND_URL = "http://localhost:8000/graph/expand"
SLOW_QUERY_LOG_PATH = Path("diagnostics/slow_queries.log")

# --- INIZIALIZZAZIONE STATO SESSIONE ---
if "messages" not in st.session_state:
    st.session_state.messages = []

if "api_status" not in st.session_state:
    st.session_state.api_status = "unknown"

if "favorite_queries" not in st.session_state:
    st.session_state.favorite_queries = []

if "conversation_stats" not in st.session_state:
    st.session_state.conversation_stats = {
        "total_queries": 0,
        "successful_queries": 0,
        "start_time": datetime.now().isoformat(),
    }

if "user_id" not in st.session_state:
    st.session_state.user_id = f"user-{datetime.now().strftime('%H%M%S')}"

if "submitted_feedback" not in st.session_state:
    st.session_state.submitted_feedback = []

# Flag di visualizzazione (valori di default, sovrascritti dalla sidebar)
debug_mode = False
show_context = False
show_query = False
show_graph = True
show_table = True
show_suggestions = True


# --- FUNZIONI UTILITY ---
def check_api_health():
    """Verifica lo stato dell'API e aggiorna lo stato della sessione."""
    try:
        response = requests.get(HEALTH_URL, timeout=5)
        if response.status_code == 200:
            st.session_state.api_status = "online"
            return True
        else:
            st.session_state.api_status = "error"
            return False
    except:
        st.session_state.api_status = "offline"
        return False


def export_conversation_json():
    """Esporta la conversazione in formato JSON."""
    data = {
        "export_date": datetime.now().isoformat(),
        "stats": st.session_state.conversation_stats,
        "messages": st.session_state.messages,
    }
    return json.dumps(data, indent=2, ensure_ascii=False)


def export_conversation_txt():
    """Esporta la conversazione in formato testo."""
    lines = [
        "=" * 60,
        "CONVERSAZIONE BLUEAI",
        f"Esportata: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}",
        "=" * 60,
        "",
    ]

    for msg in st.session_state.messages:
        role = "UTENTE" if msg["role"] == "user" else "BLUEAI"
        timestamp = msg.get("timestamp", "N/A")
        lines.append(f"\n[{role}] - {timestamp}")
        lines.append("-" * 60)
        lines.append(msg["content"])

        if msg["role"] == "assistant" and msg.get("query"):
            lines.append("\nQuery Cypher:")
            lines.append(msg["query"])
        lines.append("")

    return "\n".join(lines)


def clear_conversation():
    """Cancella la conversazione corrente."""
    st.session_state.messages = []
    st.session_state.conversation_stats = {
        "total_queries": 0,
        "successful_queries": 0,
        "start_time": datetime.now().isoformat(),
    }
    st.session_state.submitted_feedback = []
    try:
        requests.delete(f"{CONVERSATION_URL}/{st.session_state.user_id}", timeout=5)
    except Exception as exc:
        logger.warning(f"Impossibile cancellare la memoria lato server: {exc}")


def add_to_favorites(query):
    """Aggiunge una query ai preferiti."""
    if query not in st.session_state.favorite_queries:
        st.session_state.favorite_queries.append(query)
        return True
    return False


def suggest_related_queries(last_response):
    """Suggerisce query correlate basate sull'ultima risposta."""
    suggestions = []

    if "cliente" in last_response.lower() or "clienti" in last_response.lower():
        suggestions = [
            "Mostra il fatturato di questo cliente",
            "Quali sono gli altri clienti?",
            "Analizza i dettagli del cliente",
        ]
    elif "fatturato" in last_response.lower():
        suggestions = [
            "Confronta con il fatturato totale",
            "Mostra i clienti con fatturato simile",
            "Analisi trend fatturato",
        ]
    elif "ditta" in last_response.lower() or "ditte" in last_response.lower():
        suggestions = [
            "Quanti clienti ha ogni ditta?",
            "Confronta le ditte per fatturato",
            "Dettagli sulla ditta principale",
        ]
    else:
        suggestions = [
            "Mostra tutti i clienti",
            "Qual è il fatturato totale?",
            "Lista delle ditte",
        ]

    return suggestions[:3]


def render_graph(graph_data: Dict[str, Any], element_id: Optional[str] = None):
    """Mostra un grafo interattivo con pan/zoom, controlli e legenda automatica."""
    if not graph_data:
        st.info("Nessun grafo da visualizzare")
        return

    nodes = graph_data.get("nodes") or []
    edges = graph_data.get("edges") or []

    if not nodes:
        st.info("La query non ha restituito nodi visualizzabili")
        return

    nodes_json = json.dumps(nodes, ensure_ascii=False)
    edges_json = json.dumps(edges, ensure_ascii=False)
    meta_json = json.dumps(graph_data.get("meta") or {}, ensure_ascii=False)
    element_id = element_id or f"neo4j-graph-{uuid.uuid4().hex[:8]}"

    html = f"""
    <div id="{element_id}" style="width:100%;height:640px;position:relative;">

      <!-- CONTROLLI ZOOM -->
      <div style="position:absolute; right:12px; top:12px; z-index:1001; display:flex; gap:8px;">
        <button id="{element_id}-zoom-in"  class="btn">＋</button>
        <button id="{element_id}-zoom-out" class="btn">－</button>
        <button id="{element_id}-fit"      class="btn">Adatta</button>
        <button id="{element_id}-reset"    class="btn">Reset</button>
        <button id="{element_id}-back"     class="btn">Indietro</button>
        <button id="{element_id}-freeze"   class="btn">Freeze</button>
        <button id="{element_id}-restore"  class="btn">Ripristina</button>
      </div>

      <!-- TOOLTIP -->
      <div id="{element_id}-tooltip" style="
          position:absolute; display:none; background:rgba(15,23,42,0.95); color:#e2e8f0;
          padding:12px 16px; border-radius:8px; font-size:13px; pointer-events:none;
          z-index:1000; border:1px solid #334155; box-shadow:0 4px 12px rgba(0,0,0,0.3);
          max-width:300px; font-family:'Inter', sans-serif;"></div>

      <!-- SVG -->
      <svg id="{element_id}-svg"></svg>

      <!-- LEGENDA -->
      <div id="{element_id}-legend" style="
          position:absolute; top:12px; left:12px;
          background:rgba(15,23,42,0.85); border:1px solid #334155;
          border-radius:8px; padding:8px 12px; color:#e2e8f0;
          font-size:12px; font-family:'Inter',sans-serif;
          z-index:1001; max-width:300px;"></div>

      <!-- INFO PANEL -->
      <div id="{element_id}-info" class="graph-info-panel" style="
          position:absolute; bottom:12px; right:12px;
          width:340px; max-height:70%; overflow:auto;
          background:rgba(10,16,28,0.92); border:1px solid rgba(59,130,246,0.35);
          border-radius:12px; padding:14px 18px; color:#e2e8f0;
          font-size:12px; font-family:'Inter',sans-serif;
          z-index:1002; display:none; box-shadow:0 18px 38px rgba(0,0,0,0.35);
        "></div>

    </div>

    <script src="https://d3js.org/d3.v7.min.js"></script>
    <script>
    (function() {{
      const rawNodes = {nodes_json};
      const rawEdges = {edges_json};
      const expandEndpoint = "{GRAPH_EXPAND_URL}";
      const width = 780, height = 580;

      const svg = d3.select("#{element_id}-svg")
        .attr("viewBox", [0, 0, width, height])
        .style("width","100%")
        .style("height","100%")
        .style("background","#0b1220")
        .style("border-radius","12px");

      const viewport = svg.append("g").attr("class", "viewport");
      viewport.append("rect")
        .attr("x",-5000).attr("y",-5000).attr("width",10000).attr("height",10000)
        .attr("fill","transparent").style("cursor","grab")
        .on("click", () => {{
          clearSelection();
          tooltip.style('display','none');
        }});

      const linkGroup = viewport.append("g").attr("class","links");
      const nodeGroup = viewport.append("g").attr("class","nodes");
      const labelGroup = viewport.append("g").attr("class","labels");
      const edgeLabelGroup = viewport.append("g").attr("class","edge-labels");

      const tooltip = d3.select("#{element_id}-tooltip");
      const infoPanel = d3.select("#{element_id}-info");
      infoPanel.on("click", ev => ev.stopPropagation());
      const color = d3.scaleOrdinal(d3.schemeCategory10);
      const meta = {meta_json};
      const legend = d3.select("#{element_id}-legend");
      const container = document.getElementById("{element_id}");
      const edgeKey = edge => {{
        if(!edge) return "";
        const src = edge.source && edge.source.id ? edge.source.id : edge.source;
        const tgt = edge.target && edge.target.id ? edge.target.id : edge.target;
        const rel = edge.type || "REL";
        return `${{src}}->${{tgt}}:${{rel}}`;
      }};

      const zoomInBtn = document.getElementById("{element_id}-zoom-in");
      const zoomOutBtn = document.getElementById("{element_id}-zoom-out");
      const fitBtn = document.getElementById("{element_id}-fit");
      const resetBtn = document.getElementById("{element_id}-reset");
      const backBtn = document.getElementById("{element_id}-back");
      const freezeBtn = document.getElementById("{element_id}-freeze");
      const restoreBtn = document.getElementById("{element_id}-restore");
      let layoutFrozen = false;

      let selectedNode = null;
      let selectedEdge = null;

      function escapeHtml(value) {{
        return String(value)
          .replace(/&/g,"&amp;")
          .replace(/</g,"&lt;")
          .replace(/>/g,"&gt;")
          .replace(/\"/g,"&quot;")
          .replace(/'/g,"&#39;");
      }}

      function formatValue(value) {{
        if (value === null || value === undefined || value === "") return "-";
        if (Array.isArray(value)) {{
          if (!value.length) return "-";
          return value
            .map(v => (typeof v === "object" ? JSON.stringify(v) : String(v)))
            .join(", ");
        }}
        if (typeof value === "object") {{
          try {{ return JSON.stringify(value); }}
          catch (err) {{ return "[oggetto]"; }}
        }}
        return String(value);
      }}

      function buildRowsFromObject(obj) {{
        if(!obj) return [];
        return Object.keys(obj)
          .sort()
          .map(key => ({{
            label: key,
            value: formatValue(obj[key])
          }}));
      }}

      function showTooltip(event, nodeData) {{
        if(!nodeData) return;
        const rows = buildRowsFromObject(nodeData.properties || {{}}).slice(0,6);
        let html = `<div style='font-weight:600;margin-bottom:4px;'>${{escapeHtml(getNodeLabel(nodeData))}}</div>`;
        if(rows.length) {{
          html += rows
            .map(row => `<div><span style='color:#94a3b8'>${{escapeHtml(row.label)}}:</span> ${{escapeHtml(row.value)}}</div>`)
            .join("");
        }} else {{
          html += "<div>Nessuna proprietà disponibile</div>";
        }}
        tooltip.html(html).style("display","block");
        moveTooltip(event);
      }}

      function moveTooltip(event) {{
        if(!event || !container) return;
        const bounds = container.getBoundingClientRect();
        const x = event.pageX - bounds.left + 12;
        const y = event.pageY - bounds.top + 12;
        tooltip.style("left", `${{x}}px`).style("top", `${{y}}px`);
      }}

      function hideTooltip() {{
        tooltip.style("display","none");
      }}

      function showInfoPanel(payload) {{
        if(!payload) {{
          infoPanel.style("display","none").html("");
          return;
        }}
        const rows = payload.rows || [];
        const rowsHtml = rows.length
          ? rows
              .map(
                row => `<div class="info-row"><span>${{escapeHtml(row.label)}}</span><span>${{escapeHtml(row.value)}}</span></div>`
              )
              .join("")
          : '<div class="info-empty">Nessuna informazione disponibile</div>';
        const subtitleHtml = payload.subtitle
          ? `<div class="info-subtitle">${{escapeHtml(payload.subtitle)}}</div>`
          : "";
        const noteHtml = payload.note
          ? `<div class="info-note">${{escapeHtml(payload.note)}}</div>`
          : "";
        infoPanel
          .style("display","block")
          .html(
            `<div class="info-title">${{escapeHtml(payload.title || "Dettagli nodo")}}</div>` +
            subtitleHtml +
            `<div class="info-block">${{rowsHtml}}</div>` +
            noteHtml
          );
      }}

      function updateFreezeButton() {{
        if (!freezeBtn) return;
        freezeBtn.textContent = layoutFrozen ? "Riprendi" : "Freeze";
        freezeBtn.classList.toggle("active", layoutFrozen);
      }}

      function toggleFreeze() {{
        layoutFrozen = !layoutFrozen;
        if (layoutFrozen) {{
          simulation.stop();
        }} else {{
          simulation.alpha(0.6).restart();
        }}
        updateFreezeButton();
      }}

      function restoreLayout() {{
        layoutFrozen = false;
        updateFreezeButton();
        nodes.forEach(n => {{
          n.fx = null;
          n.fy = null;
        }});
        if (node) {{
          node.classed("pinned", false);
        }}
        camStack.length = 0;
        pushCamState();
        clearSelection();
        simulation.alpha(0.9).restart();
        fit(false);
      }}

      function isPinned(nodeData) {{
        if (!nodeData) return false;
        return nodeData.fx != null || nodeData.fy != null;
      }}

      function togglePin(nodeData, element) {{
        if (!nodeData) return;
        const pinned = isPinned(nodeData);
        if (pinned) {{
          nodeData.fx = null;
          nodeData.fy = null;
        }} else {{
          nodeData.fx = nodeData.x;
          nodeData.fy = nodeData.y;
          if (!layoutFrozen) {{
            simulation.alphaTarget(0.3).restart();
          }}
        }}
        if (element) {{
          d3.select(element).classed("pinned", !pinned);
        }}
        if (node) {{
          node.classed("pinned", d => isPinned(d));
        }}
        if (!pinned && !layoutFrozen) {{
          simulation.alphaTarget(0);
        }}
      }}

      function focusNode(nodeData) {{
        if (!nodeData) return;
        pushCamState();
        const k = 1.5;
        const tx = width / 2 - k * (nodeData.x || 0);
        const ty = height / 2 - k * (nodeData.y || 0);
        smooth(d3.zoomIdentity.translate(tx, ty).scale(k));
      }}

      function refreshSelections() {{
        node.classed("selected", n => selectedNode && n.id === selectedNode.id);
        link.classed("selected", l => selectedEdge && edgeKey(l) === edgeKey(selectedEdge));
      }}

      function clearSelection() {{
        selectedNode = null;
        selectedEdge = null;
        refreshSelections();
        hideTooltip();
        showInfoPanel(null);
      }}

      function handleNodeClick(event, nodeData) {{
        event.stopPropagation();
        if (event.altKey || event.metaKey) {{
          focusNode(nodeData);
          return;
        }}
        if (event.shiftKey || event.ctrlKey) {{
          togglePin(nodeData, event.currentTarget);
          return;
        }}
        selectedNode = nodeData;
        selectedEdge = null;
        refreshSelections();
        showInfoPanel({{
          title: getNodeLabel(nodeData),
          subtitle: (nodeData.labels || []).join(" · "),
          rows: buildRowsFromObject(nodeData.properties || {{}})
        }});
      }}

      function handleEdgeClick(event, edgeData) {{
        event.stopPropagation();
        selectedEdge = edgeData;
        selectedNode = null;
        refreshSelections();
        const sourceLabel = (edgeData.source && edgeData.source.labels && edgeData.source.labels[0]) || "Nodo";
        const targetLabel = (edgeData.target && edgeData.target.labels && edgeData.target.labels[0]) || "Nodo";
        showInfoPanel({{
          title: edgeData.type || "REL",
          subtitle: `${{sourceLabel}} → ${{targetLabel}}`,
          rows: buildRowsFromObject(edgeData.properties || {{}})
        }});
      }}

      function mergeExpandedGraph(graphPayload, expandedNode) {{
        if(!graphPayload) return false;
        let changed = false;
        const incomingNodes = (graphPayload.nodes || []).map(n => ({{...n, id:String(n.id)}}));
        incomingNodes.forEach(n => {{
          if(!nodeMap.has(n.id)) {{
            nodes.push(n);
            nodeMap.set(n.id, n);
            changed = true;
          }} else {{
            const existing = nodeMap.get(n.id);
            existing.labels = n.labels || existing.labels;
            existing.properties = {{...(existing.properties || {{}}), ...(n.properties || {{}})}};
          }}
        }});

        const existingEdgeIds = new Set(links.map(edgeKey));
        const pivot = (meta && meta.pivot) ? meta.pivot : null;
        (graphPayload.edges || []).forEach(e => {{
          const sourceNode = nodeMap.get(String(e.source));
          const targetNode = nodeMap.get(String(e.target));
          if(!sourceNode || !targetNode) return;
          const enriched = {{
            ...e,
            id: e.id ? String(e.id) : `${{sourceNode.id}}->${{targetNode.id}}:${{e.type || "REL"}}`,
            source: sourceNode,
            target: targetNode,
          }};
          if (pivot && enriched.type === 'APPARTIENE_A') {{
            const pid = String(pivot.id || '');
            if (enriched.source.id !== pid && enriched.target.id !== pid) {{
              return; // evita collegamenti verso altre ditte per coerenza visiva
            }}
          }}
          const key = edgeKey(enriched);
          if(existingEdgeIds.has(key)) return;
          existingEdgeIds.add(key);
          const withMark = expandedNode ? {{...enriched, _addedBy: expandedNode.id}} : enriched;
          links.push(withMark);
          changed = true;
        }});

        if(expandedNode) {{
          expandedNode._expanded = true;
        }}
        return changed;
      }}

      async function expandNode(event, nodeData) {{
        event.stopPropagation();
        if(!nodeData || !nodeData.id) return;
        // Toggle collapse se già espanso: rimuovi ciò che è stato aggiunto da questo nodo
        if (nodeData._expanded) {{
          for (let i = links.length - 1; i >= 0; i--) {{
            if (links[i]._addedBy === nodeData.id) links.splice(i,1);
          }}
          // eventuale rimozione nodi aggiunti (solo se non hanno altre connessioni)
          const removedNodeIds = new Set();
          // per semplicità non rimuoviamo nodi, evitiamo inconsistenze grafiche
          nodeData._expanded = false;
          updateGraph(true);
          handleNodeClick({{ stopPropagation: () => {{}} }}, nodeData);
          return;
        }}
        if(nodeData._expanding) return;
        nodeData._expanding = true;
        showInfoPanel({{
          title: getNodeLabel(nodeData),
          subtitle: "Espansione vicini",
          note: "Recupero del vicinato in corso..."
        }});
        try {{
          const response = await fetch(expandEndpoint, {{
            method: "POST",
            headers: {{"Content-Type":"application/json"}},
            body: JSON.stringify({{node_id: nodeData.id, limit: 25}})
          }});
          if(!response.ok) {{
            throw new Error("Espansione non disponibile");
          }}
          const payload = await response.json();
          const graphPayload = (payload && payload.graph_data) || {{}};
          const changed = mergeExpandedGraph(graphPayload, nodeData);
          if(changed) {{
            updateGraph(true);
            handleNodeClick({{ stopPropagation: () => {{}} }}, nodeData);
          }} else {{
            showInfoPanel({{
              title: getNodeLabel(nodeData),
              subtitle: "Espansione vicini",
              note: "Nessun vicino trovato."
            }});
          }}
        }} catch (err) {{
          console.error("Errore espansione nodo", err);
          showInfoPanel({{
            title: "Errore espansione",
            note: err && err.message ? err.message : "Impossibile espandere il nodo."
          }});
        }} finally {{
          nodeData._expanding = false;
        }}
      }}

      // --- NODI E ARCHI ---
      const nodes = rawNodes.map(n => ({{...n, id:String(n.id)}}));
      const nodeMap = new Map(nodes.map(n => [n.id, n]));
      const links = rawEdges.map(e => {{
        const sourceId = String(e.source);
        const targetId = String(e.target);
        const sourceNode = nodeMap.get(sourceId);
        const targetNode = nodeMap.get(targetId);
        if(!sourceNode || !targetNode) return null;
        const edgeId = e.id ? String(e.id) : `${{sourceId}}->${{targetId}}:${{e.type || "REL"}}`;
        return {{...e, id: edgeId, source: sourceNode, target: targetNode}};
      }}).filter(Boolean);

      const zoom = d3
        .zoom()
        .scaleExtent([0.1, 4])
        .on("zoom", event => viewport.attr("transform", event.transform));
      svg.call(zoom).on("dblclick.zoom", null);

      const camStack = [];
      function pushCamState() {{
        const current = d3.zoomTransform(svg.node());
        camStack.push(current);
        if (camStack.length > 50) {{
          camStack.shift();
        }}
      }}
      function popCamState() {{
        return camStack.pop();
      }}
      function smooth(transform) {{
        svg.transition().duration(300).call(zoom.transform, transform);
      }}
      function zoomBy(factor) {{
        pushCamState();
        svg.transition().duration(250).call(zoom.scaleBy, factor);
      }}
      function resetView() {{
        pushCamState();
        smooth(d3.zoomIdentity);
      }}
      function fit(pushState = true) {{
        if (!nodes.length) {{
          resetView();
          return;
        }}
        const xs = nodes.map(n => n.x || 0);
        const ys = nodes.map(n => n.y || 0);
        const minX = Math.min(...xs);
        const maxX = Math.max(...xs);
        const minY = Math.min(...ys);
        const maxY = Math.max(...ys);
        const padding = 40;
        const boundsWidth = (maxX - minX) || 1;
        const boundsHeight = (maxY - minY) || 1;
        const scale = Math.min(
          (width - padding) / boundsWidth,
          (height - padding) / boundsHeight,
          4
        );
        const tx = width / 2 - scale * (minX + maxX) / 2;
        const ty = height / 2 - scale * (minY + maxY) / 2;
        const transform = d3.zoomIdentity.translate(tx, ty).scale(scale);
        if (pushState) {{
          pushCamState();
        }}
        smooth(transform);
      }}
      function goBack() {{
        const previous = popCamState();
        if (previous) {{
          smooth(previous);
        }}
      }}
      pushCamState();

      // --- SIMULAZIONE ---
      const linkForce = d3.forceLink(links)
        .id(d=>d.id)
        .distance(140)
        .strength(0.25);

      const simulation = d3.forceSimulation(nodes)
        .force("link", linkForce)
        .force("charge", d3.forceManyBody().strength(-120))
        .force("center", d3.forceCenter(width/2, height/2))
        .force("collision", d3.forceCollide().radius(34).strength(0.85))
        .velocityDecay(0.36)
        .alphaDecay(0.12);
      simulation.on("tick", ticked);

      // --- FRECCE ---
      svg.append("defs").append("marker")
        .attr("id","arrowhead").attr("viewBox","0 -5 10 10")
        .attr("refX",28).attr("refY",0)
        .attr("markerWidth",6).attr("markerHeight",6)
        .attr("orient","auto")
        .append("path").attr("d","M0,-5L10,0L0,5").attr("fill","#475569");

      const drag = d3.drag()
        .on("start",(ev,d)=>{{if(!ev.active)simulation.alphaTarget(0.3).restart();d.fx=d.x;d.fy=d.y;}})
        .on("drag",(ev,d)=>{{d.fx=ev.x;d.fy=ev.y;}})
        .on("end",(ev,d)=>{{if(!ev.active)simulation.alphaTarget(0);d.fx=null;d.fy=null;}});

      let link = linkGroup.selectAll("line");
      let edgeLabel = edgeLabelGroup.selectAll("text");
      let node = nodeGroup.selectAll("circle");
      let label = labelGroup.selectAll("text");

      function ticked() {{
        link
          .attr("x1", d => d.source.x)
          .attr("y1", d => d.source.y)
          .attr("x2", d => d.target.x)
          .attr("y2", d => d.target.y);
        node
          .attr("cx", d => d.x)
          .attr("cy", d => d.y);
        label
          .attr("x", d => d.x)
          .attr("y", d => d.y - 28);
        edgeLabel
          .attr("x", d => (d.source.x + d.target.x) / 2)
          .attr("y", d => (d.source.y + d.target.y) / 2 - 8);
      }}

      function getNodeLabel(d){{
        const p=d.properties||{{}};
        return p.nome||p.name||p.title||
               (p.descrizione?String(p.descrizione).substring(0,15):null)||
               (d.labels&&d.labels[0])||String(d.id).substring(0,8);
      }}

      // Filtri per etichette (visibilità) applicati lato client
      let visibleLabels = new Set();
      function applyVisibilityFilters() {{
        node.style("display", d => {{
          const lbl = (d.labels && d.labels[0]) || 'Node';
          return visibleLabels.has(lbl) ? null : 'none';
        }});
        label.style("display", d => {{
          const lbl = (d.labels && d.labels[0]) || 'Node';
          return visibleLabels.has(lbl) ? null : 'none';
        }});
        link.style("display", l => {{
          const sLbl = (l.source && l.source.labels && l.source.labels[0]) || 'Node';
          const tLbl = (l.target && l.target.labels && l.target.labels[0]) || 'Node';
          return (visibleLabels.has(sLbl) && visibleLabels.has(tLbl)) ? null : 'none';
        }});
        edgeLabel.style("display", l => {{
          const sLbl = (l.source && l.source.labels && l.source.labels[0]) || 'Node';
          const tLbl = (l.target && l.target.labels && l.target.labels[0]) || 'Node';
          return (visibleLabels.has(sLbl) && visibleLabels.has(tLbl)) ? null : 'none';
        }});
      }}

      function updateLegend() {{
        const uniqueLabels = Array.from(new Set(nodes.flatMap(n => n.labels || ["Node"])));
        const resultCount = (meta && meta.result_node_ids && meta.result_node_ids.length)
          ? meta.result_node_ids.length
          : nodes.filter(n => n.isResult).length;
        const nodeCount = nodes.length;
        const linkCount = links.length;
        const resultsPart = resultCount ? (' · Risultati: ' + String(resultCount)) : '';
        let head = '';
        head += "<div style='margin-bottom:6px;'>";
        head += "<strong style='color:#38bdf8; font-size: 1.05rem;'>Legenda</strong>";
        head += "<div style='color:#cbd5e1; font-size:0.9rem;'>" +
                "Nodi: " + String(nodeCount) +
                " · Relazioni: " + String(linkCount) +
                resultsPart +
                "</div>";
        head += "</div>";
        const labelsHtml = uniqueLabels.map(lbl => {{
          const col = color(lbl);
          return (
            "<span style='display:inline-flex;align-items:center;margin-right:10px; font-size: 0.95rem;'>" +
            "<span style='width:10px;height:10px;background:" + col + ";" +
            "border-radius:50%;display:inline-block;margin-right:6px;border:1px solid #334155;'></span>" +
            String(lbl) +
            "</span>"
          );
        }}).join("<br/>");
        if (visibleLabels.size === 0) {{ uniqueLabels.forEach(l => visibleLabels.add(l)); }}
        const filtersHtml = uniqueLabels.map(lbl => {{
          const checked = visibleLabels.has(lbl) ? 'checked' : '';
          return (
            "<label style='display:flex;align-items:center;gap:6px;margin-top:4px;'>" +
            "<input class='legend-filter' type='checkbox' data-label='" + String(lbl) + "' " + checked + " />" +
            "<span>" + String(lbl) + "</span>" +
            "</label>"
          );
        }}).join("");
        let legendHtml = head + labelsHtml +
          "<hr style='border-color:rgba(59,130,246,0.15);margin:6px 0;' />" +
          "<div style='font-size:0.85rem;color:#cbd5e1;margin-bottom:4px;'>Filtri etichette</div>" +
          filtersHtml;
        if(meta && meta.stub_nodes_added) {{
          legendHtml += "<hr style='border-color:rgba(59,130,246,0.35);margin:6px 0;' />";
          legendHtml += "<span style='display:flex;align-items:center;gap:8px;font-size:0.9rem;'><span style='width:12px;height:12px;border-radius:50%;border:2px dashed rgba(148,163,184,0.9);'></span>Segnaposto (nessun nodo Neo4j trovato)</span>";
        }}
        legendHtml += "<hr style='border-color:rgba(59,130,246,0.15);margin:6px 0;' />";
        legendHtml += "<span style='font-size:0.85rem;color:#94a3b8;'>Clic: dettagli · Alt/⌘: focus · Shift/Ctrl: blocca · Doppio click: espandi</span>";
        legend.html(legendHtml);
        d3.selectAll(`#${element_id}-legend input.legend-filter`).on('change', function(){{
          const lbl = this.getAttribute('data-label');
          if (this.checked) visibleLabels.add(lbl); else visibleLabels.delete(lbl);
          applyVisibilityFilters();
        }});
        applyVisibilityFilters();
      }}

      function updateGraph(restart=true){{
        nodeMap.clear();
        nodes.forEach(n => nodeMap.set(n.id, n));
        links.forEach(l => {{
          if(l && typeof l.source === "string"){{
            const src=nodeMap.get(String(l.source));
            if(src) l.source=src;
          }}
          if(l && typeof l.target === "string"){{
            const tgt=nodeMap.get(String(l.target));
            if(tgt) l.target=tgt;
          }}
        }});

        link = linkGroup.selectAll("line")
          .data(links, edgeKey);
        link.exit().remove();
        const linkEnter = link.enter().append("line")
          .attr("stroke","#475569").attr("stroke-opacity",0.65)
          .attr("stroke-width",2).attr("marker-end","url(#arrowhead)")
          .style("cursor","pointer")
          .on("click", handleEdgeClick);
        link = linkEnter.merge(link)
          .classed("selected", l => selectedEdge && edgeKey(l) === edgeKey(selectedEdge));

        edgeLabel = edgeLabelGroup.selectAll("text")
          .data(links, edgeKey);
        edgeLabel.exit().remove();
        const edgeLabelEnter = edgeLabel.enter().append("text")
          .attr("fill","#94a3b8").attr("font-size",10)
          .attr("text-anchor","middle").attr("font-family","Inter, monospace")
          .attr("font-weight","500");
        edgeLabel = edgeLabelEnter.merge(edgeLabel)
          .text(d => d.type || "REL");

        node = nodeGroup.selectAll("circle").data(nodes, d => d.id);
        node.exit().remove();
        const nodeEnter = node.enter().append("circle")
          .attr("r",20)
          .attr("stroke","#0f172a").attr("stroke-width",2)
          .style("cursor","pointer")
          .call(drag)
          .on("mouseover", showTooltip)
          .on("mousemove", moveTooltip)
          .on("mouseout", hideTooltip)
          .on("click", handleNodeClick)
          .on("dblclick", expandNode);
        node = nodeEnter.merge(node)
          .attr("fill", d => color((d.labels && d.labels[0]) || "Node"))
          .classed("result-node", d => !!d.isResult)
          .classed("expanded-node", d => !!d._expanded)
          .classed("selected", n => selectedNode && n.id === selectedNode.id)
          .classed("pinned", d => isPinned(d));

        label = labelGroup.selectAll("text").data(nodes, d => d.id);
        label.exit().remove();
        const labelEnter = label.enter().append("text")
          .attr("fill","#f1f5f9").attr("font-size",12).attr("text-anchor","middle")
          .attr("font-family","Inter, sans-serif").attr("font-weight","600")
          .attr("pointer-events","none");
        label = labelEnter.merge(label)
          .text(getNodeLabel);

        simulation.nodes(nodes);
        linkForce.links(links);
        if(restart){{
          simulation.alpha(0.9).restart();
        }}

        updateLegend();
        refreshSelections();
      }}

      updateGraph(true);

      if (zoomInBtn) zoomInBtn.onclick = () => zoomBy(1.2);
      if (zoomOutBtn) zoomOutBtn.onclick = () => zoomBy(1 / 1.2);
      if (resetBtn) resetBtn.onclick = () => resetView();
      if (fitBtn) fitBtn.onclick = () => fit();
      if (backBtn) backBtn.onclick = () => goBack();
      if (freezeBtn) freezeBtn.onclick = () => toggleFreeze();
      if (restoreBtn) restoreBtn.onclick = () => restoreLayout();
      updateFreezeButton();
      simulation.on("end", () => fit(false));
      setTimeout(() => fit(false), 600);

    }})();
    </script>
        <style>
      #{element_id} .btn {{
        background: rgba(15,23,42,0.85);
        color: #cbd5f5;
        border: 1px solid rgba(59,130,246,0.35);
        border-radius: 8px;
        padding: 6px 10px;
        font-size: 13px;
        cursor: pointer;
        transition: all 0.2s ease;
      }}
      #{element_id} .btn:hover {{
        background: rgba(59,130,246,0.25);
        color: #e0ecff;
      }}
      #{element_id} .btn.active {{
        background: rgba(59,130,246,0.35);
        color: #f8fafc;
        border-color: rgba(148,197,255,0.9);
        box-shadow: 0 0 12px rgba(59,130,246,0.3);
      }}
      #{element_id} .nodes circle {{
        transition: stroke-width 0.2s ease;
      }}
      #{element_id} .nodes circle:hover {{
        stroke-width: 2.4;
      }}
      #{element_id} .nodes circle.result-node {{
        stroke: #38bdf8;
        stroke-width: 3;
        filter: drop-shadow(0 0 8px rgba(56,189,248,0.4));
      }}
      #{element_id} .nodes circle.pinned {{
        stroke: #facc15;
        stroke-width: 3;
        filter: drop-shadow(0 0 8px rgba(250,204,21,0.35));
      }}
      #{element_id} .nodes circle.selected {{
        stroke: #f97316;
        stroke-width: 3;
        filter: drop-shadow(0 0 10px rgba(249,115,22,0.45));
      }}
      #{element_id} .links line.selected {{
        stroke: #f97316;
        stroke-width: 2.4;
      }}
      #{element_id} .graph-info-panel .info-title {{
        font-weight: 700;
        font-size: 1.1rem;
        margin-bottom: 4px;
        color: #f8fafc;
      }}
      #{element_id} .graph-info-panel .info-subtitle {{
        color: #93c5fd;
        font-size: 0.9rem;
        letter-spacing: 0.5px;
        text-transform: uppercase;
        margin-bottom: 10px;
      }}
      #{element_id} .graph-info-panel .info-block {{
        border-top: 1px solid rgba(59,130,246,0.2);
        margin-top: 10px;
        padding-top: 8px;
      }}
      #{element_id} .graph-info-panel .info-row {{
        display: flex;
        justify-content: space-between;
        gap: 12px;
        font-size: 0.88rem;
        padding: 2px 0;
        color: #e2e8f0;
      }}
      #{element_id} .graph-info-panel .info-row span:first-child {{
        color: #94a3b8;
        font-weight: 500;
      }}
      #{element_id} .graph-info-panel .info-note {{
        background: rgba(59,130,246,0.12);
        border: 1px solid rgba(59,130,246,0.3);
        border-radius: 8px;
        padding: 8px;
        margin-bottom: 10px;
        font-size: 0.85rem;
        color: #dbeafe;
      }}
      #{element_id} .graph-info-panel .info-empty {{
        font-style: italic;
        color: #94a3b8;
      }}
    </style>
    """

    components.html(html, height=660)


def render_table(context) -> bool:
    """Mostra i risultati in una tabella stilizzata (molto più carina)."""
    if not context or not isinstance(context, list):
        return False

    rows = []
    for item in context:
        if isinstance(item, dict):
            row = {}
            for key, value in item.items():
                if isinstance(value, (dict, list)):
                    row[key] = json.dumps(value, ensure_ascii=False)
                else:
                    row[key] = value
            if row:
                rows.append(row)

    if not rows:
        return False

    # --- MODIFICHE QUI ---

    # 1. Converti la tua lista di dict in un DataFrame Pandas
    df = pd.DataFrame(rows)

    # 2. Crea un oggetto "Styler" per definire la formattazione
    styler = (
        df.style.hide(axis="index")
        .format(precision=2, na_rep="-")
        .set_properties(**{"text-align": "left"})
        .set_table_styles(
            [
                dict(
                    selector="th",
                    props=[("text-align", "left"), ("font-weight", "bold")],
                ),
                dict(selector="tr:hover", props=[("background-color", "#f5f5f5")]),
            ]
        )
    )

    # 3. Mostra la tabella stilizzata in HTML
    # (Questo è il modo migliore per assicurarsi che tutti gli stili vengano applicati)
    # st.write(styler.to_html(), unsafe_allow_html=True)

    # In alternativa, se vuoi ancora la griglia interattiva ma con stili:
    st.dataframe(styler, width="stretch")

    # -----------------------

    return True


def render_execution_details_ui(
    timing_details: Optional[Dict[str, Any]],
    label: str = "Dettagli esecuzione",
) -> None:
    """Mostra un riepilogo compatto dei metadati di esecuzione."""

    if not timing_details:
        return

    complexity = timing_details.get("query_complexity") or {}
    slow = timing_details.get("slow_query")
    timeout_val = timing_details.get("query_timeout_seconds")

    with st.expander(f"📊 {label}", expanded=False):
        col1, col2 = st.columns(2)
        level = complexity.get("level")
        col1.metric("Complessità", level.upper() if level else "-")
        if timeout_val:
            col2.metric("Timeout applicato", f"{timeout_val:.1f}s")
        else:
            col2.metric("Timeout applicato", "default")

        reasons = complexity.get("reasons") or []
        if reasons:
            st.markdown(
                "**Motivi rilevati:** "
                + ", ".join(sorted(set(str(reason) for reason in reasons)))
            )

        safe_info = timing_details.get("safe_rewrite") or {}
        if safe_info.get("applied"):
            notes = safe_info.get("notes") or []
            text = "; ".join(notes) if notes else "riscrittura di sicurezza applicata"
            st.info(f"Rewrite di sicurezza attivo: {text}")

        attempts = timing_details.get("execution_attempts") or []
        if attempts:
            df_attempts = pd.DataFrame(attempts)
            if "query" in df_attempts.columns:
                df_attempts["query"] = df_attempts["query"].apply(
                    lambda q: (
                        (q[:180] + "…") if isinstance(q, str) and len(q) > 200 else q
                    )
                )
            st.markdown("**Tentativi di esecuzione**")
            st.dataframe(df_attempts, width="stretch")

        semantic_iters = timing_details.get("semantic_expansion_iterations")
        if semantic_iters:
            st.caption(f"Tentativi di espansione semantica: {semantic_iters}")

        examples_similarity = timing_details.get("examples_similarity")
        if isinstance(examples_similarity, (int, float)):
            st.caption(f"Similarità esempio selezionato: {examples_similarity:.3f}")

        if slow:
            st.warning(slow.get("message", "Query interrotta per eccessiva durata."))


def show_feedback_form(
    user_question: Optional[str],
    answer: Optional[str],
    query: Optional[str],
    timing_details: Optional[Dict[str, Any]],
    key: str,
) -> None:
    """Mostra il form di feedback collegato al nuovo endpoint backend."""

    if not FEEDBACK_URL:
        return

    user_question = user_question or ""
    submitted_keys = set(st.session_state.submitted_feedback)
    already_sent = key in submitted_keys

    labels = {
        "corretta": "✅ Corretta",
        "incompleta": "⚠️ Incompleta",
        "fuori_fuoco": "🎯 Fuori fuoco",
        "troppo_formale": "🗣️ Tonalità da rivedere",
        "troppo_lunga": "📏 Troppo lunga",
    }

    with st.expander("🗳️ Feedback", expanded=False):
        with st.form(key=f"feedback_form_{key}"):
            category = st.selectbox(
                "Valutazione",
                options=list(labels.keys()),
                format_func=lambda c: labels.get(c, c),
                disabled=already_sent,
            )
            notes = st.text_area(
                "Note (opzionali)",
                disabled=already_sent,
                placeholder="Spiega cosa dovremmo migliorare oppure conferma che va bene.",
            )
            submitted = st.form_submit_button(
                "Invia feedback",
                disabled=already_sent,
                width="stretch",
            )

            if submitted:
                payload = {
                    "user_id": st.session_state.user_id,
                    "question": user_question,
                    "category": category,
                    "notes": notes or None,
                    "answer": answer,
                    "query_generated": query,
                    "metadata": {"timing_details": timing_details},
                }
                try:
                    resp = requests.post(FEEDBACK_URL, json=payload, timeout=10)
                    if resp.status_code == 200:
                        st.success("Feedback registrato, grazie!")
                        st.session_state.submitted_feedback.append(key)
                    else:
                        st.error(
                            f"Errore durante l'invio del feedback ({resp.status_code})."
                        )
                except Exception as exc:
                    st.error(f"Impossibile inviare il feedback: {exc}")

        if already_sent:
            st.caption("Feedback già inviato per questa risposta.")


def load_slow_query_log(limit: int = 25) -> List[Dict[str, Any]]:
    """Carica gli ultimi slow query log dal file JSONL."""

    if not SLOW_QUERY_LOG_PATH.exists():
        return []
    entries: List[Dict[str, Any]] = []
    try:
        lines = SLOW_QUERY_LOG_PATH.read_text(encoding="utf-8").splitlines()
        for line in lines[-limit:]:
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    except Exception as exc:
        logger.warning(f"Impossibile leggere slow query log: {exc}")
    return entries


# --- FUNZIONE PRINCIPALE PER PROCESSARE QUERY ---
def process_query(prompt):
    """Processa una query e gestisce la risposta dall'API."""

    # Aggiorna statistiche
    st.session_state.conversation_stats["total_queries"] += 1

    # Aggiungi il messaggio utente
    user_message = {
        "role": "user",
        "content": prompt,
        "timestamp": datetime.now().isoformat(),
    }
    st.session_state.messages.append(user_message)

    # Mostra il messaggio utente
    with st.chat_message("user"):
        st.markdown(prompt)

    # Prepara il messaggio assistente
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        status_placeholder = st.empty()

        try:
            # Mostra stato caricamento
            with status_placeholder:
                st.info(" BlueAI sta elaborando la tua richiesta...")

            logger.info(f" Query: {prompt}")

            # Chiamata API
            payload = {"question": prompt, "user_id": st.session_state.user_id}
            response = requests.post(API_URL, json=payload, timeout=300)

            status_placeholder.empty()

            if response.status_code == 200:
                data = response.json()
                logger.info(f" Risposta ricevuta")

                # Estrai i dati
                risposta_ai = data.get(
                    "risposta", "Non ho ricevuto una risposta valida."
                )
                query_generata = data.get("query_generata", "")
                context = data.get("context", [])
                graph_data = data.get("graph_data") or {}
                success = data.get("success", False)
                timing_details = data.get("timing_details", {})
                graph_origin = timing_details.get("graph_origin")
                message_id = str(uuid.uuid4())

                # Aggiorna statistiche
                if success:
                    st.session_state.conversation_stats["successful_queries"] += 1

                # Mostra la risposta
                message_placeholder.markdown(risposta_ai)
                slow_meta = timing_details.get("slow_query")
                if slow_meta and isinstance(slow_meta, dict):
                    st.warning(
                        slow_meta.get(
                            "message", "Query interrotta per eccessiva durata."
                        )
                    )

                # Crea il messaggio da salvare
                assistant_message = {
                    "role": "assistant",
                    "content": risposta_ai,
                    "query": (
                        query_generata
                        if query_generata and query_generata != "N/D"
                        else None
                    ),
                    "context": context if context else None,
                    "graph_data": graph_data if graph_data else None,
                    "success": success,
                    "timestamp": datetime.now().isoformat(),
                    "timing_details": timing_details,
                    "message_id": message_id,
                    "metadata": {
                        "slow_query": slow_meta,
                        "query_complexity": timing_details.get("query_complexity"),
                        "safe_rewrite": timing_details.get("safe_rewrite"),
                    },
                }

                # Mostra query Cypher se presente e richiesto
                if (
                    show_query
                    and query_generata
                    and query_generata not in ["N/D", "Errore", ""]
                ):
                    st.code(query_generata, language="cypher")

                # Mostra debug info se richiesto
                if debug_mode:
                    debug_info = {
                        "success": success,
                        "query_presente": bool(
                            query_generata
                            and query_generata not in ["N/D", "Errore", ""]
                        ),
                        "context_items": len(context) if context else 0,
                        "timestamp": assistant_message["timestamp"],
                        "graph_origin": timing_details.get("graph_origin"),
                        "examples_used": data.get("examples_used"),
                        "complexity": timing_details.get("query_complexity"),
                        "safe_rewrite": timing_details.get("safe_rewrite"),
                        "execution_attempts": timing_details.get("execution_attempts"),
                    }
                    with st.expander(" Debug Info"):
                        st.json(debug_info)

                # Mostra context se richiesto e presente
                if show_context and context:
                    with st.expander(" Dati Raw dal Grafo"):
                        st.json(context)

                if show_table and context:
                    with st.expander(" Tabella risultati", expanded=False):
                        if not render_table(context):
                            st.info("Contenuto non tabellare")

                if show_graph and graph_data and graph_data.get("nodes"):
                    with st.expander(" Visualizzazione Grafo", expanded=False):
                        render_graph(graph_data)
                elif show_graph:
                    with st.expander(" Visualizzazione Grafo", expanded=False):
                        if graph_origin == "reconstructed_from_query":
                            st.info(
                                "Grafo ricostruito ma vuoto (nessun nodo distinto trovato)."
                            )
                        elif graph_origin == "context_results":
                            st.info(
                                "La query non ha restituito nodi Neo4j da visualizzare."
                            )
                        else:
                            st.info("Nessun grafo disponibile per questa risposta.")

                # Suggerimenti intelligenti
                if show_suggestions and success:
                    suggestions = suggest_related_queries(risposta_ai)
                    if suggestions:
                        st.markdown("** Domande correlate:**")
                        cols = st.columns(len(suggestions))
                        for idx, suggestion in enumerate(suggestions):
                            if cols[idx].button(
                                suggestion,
                                key=f"suggest_{datetime.now().timestamp()}_{idx}",
                            ):
                                st.session_state.pending_query = suggestion
                                st.rerun()

                # Warning se nessun dato
                if not success or "Non ho trovato" in risposta_ai:
                    st.warning(" Nessun dato trovato. Prova a riformulare la domanda.")

                render_execution_details_ui(
                    timing_details, label="Dettagli esecuzione (risposta corrente)"
                )

                show_feedback_form(
                    user_question=prompt,
                    answer=risposta_ai,
                    query=query_generata,
                    timing_details=timing_details,
                    key=message_id,
                )

                # Salva il messaggio
                st.session_state.messages.append(assistant_message)

            else:
                error_msg = f" Errore API ({response.status_code})"
                message_placeholder.error(error_msg)
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": error_msg,
                        "success": False,
                        "timestamp": datetime.now().isoformat(),
                    }
                )

        except requests.exceptions.Timeout:
            error_msg = " Timeout: L'API ha impiegato troppo tempo. Riprova."
            message_placeholder.error(error_msg)
            status_placeholder.empty()
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": error_msg,
                    "success": False,
                    "timestamp": datetime.now().isoformat(),
                }
            )

        except requests.exceptions.ConnectionError:
            error_msg = " Errore di connessione: Verifica che l'API sia attiva su http://localhost:8000"
            message_placeholder.error(error_msg)
            status_placeholder.empty()
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": error_msg,
                    "success": False,
                    "timestamp": datetime.now().isoformat(),
                }
            )

        except Exception as e:
            error_msg = f" Errore imprevisto: {str(e)}"
            message_placeholder.error(error_msg)
            status_placeholder.empty()
            logger.error(f"Errore: {e}", exc_info=True)
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": error_msg,
                    "success": False,
                    "timestamp": datetime.now().isoformat(),
                }
            )
        finally:
            if "status_placeholder" in locals():
                status_placeholder.empty()


# --- SIDEBAR ULTRA AVANZATA ---
if is_debug_mode:
    with st.sidebar:
        st.markdown("### Centro di Controllo")

        # Test connessione con badge status
        col1, col2 = st.columns([3, 1])
        with col1:
            if st.button(" Verifica API", width="stretch", key="test_api"):
                with st.spinner("Connessione..."):
                    if check_api_health():
                        st.success("✅ Online!")
                        st.balloons()
                    else:
                        st.error(" Offline")

        # with col2:
        # if st.session_state.api_status == "online":
        # st.markdown('<span class="status-badge status-online">●</span>', unsafe_allow_html=True)
        # else:
        # st.markdown('<span class="status-badge status-offline">●</span>', unsafe_allow_html=True)

        st.markdown("---")

        # Tabs per organizzare funzionalità
        tab1, tab2, tab3 = st.tabs(["Impostazioni", " Preferiti", " Stats"])

        with tab1:
            st.markdown("#### Sessione")
            st.text_input(
                "ID Utente",
                key="user_id",
                help="Personalizza l'identificativo per mantenere conversazioni distinte.",
            )
            st.markdown("---")
            st.markdown("#### Visualizzazione")
            debug_mode = st.checkbox(" Debug Mode", help="Mostra dettagli tecnici")
            show_context = st.checkbox(" Dati Raw", help="Visualizza dati dal grafo")
            show_query = st.checkbox(
                " Query Cypher", value=False, help="Mostra query generate"
            )
            show_graph = st.checkbox(
                " Visualizza Grafo",
                value=True,
                help="Mostra il sottografo estratto dalla query Cypher",
            )
            show_table = st.checkbox(
                " Tabella risultati",
                value=True,
                help="Mostra i risultati in formato tabellare quando disponibile",
            )
            show_suggestions = st.checkbox(
                " Suggerimenti Smart", value=True, help="Suggerimenti automatici"
            )

            st.markdown("---")
            st.markdown("#### Azioni Rapide")

            if st.button("Cancella Chat", width="stretch", type="secondary"):
                if st.session_state.messages:
                    clear_conversation()
                    st.success("Chat cancellata!")
                    st.rerun()
                else:
                    st.info("Chat già vuota")

            # Aggiorna il testo di aiuto per spiegare che il reset è completo
            if st.button(
                "Reset Completo del Chatbot",
                width="stretch",
                type="primary",
                help="Resetta completamente il chatbot, cancellando sia la cache delle risposte che la memoria delle conversazioni.",
            ):
                with st.spinner("Reset del server in corso..."):
                    try:
                        # La chiamata API rimane la stessa
                        response = requests.delete(CACHE_URL, timeout=10)
                        if response.status_code == 200:
                            data = response.json()

                            # Pulisci la conversazione locale
                            # Estrai i due nuovi valori dalla risposta JSON
                            risposte_rimosse = data.get("risposte_rimosse", 0)
                            sessioni_rimosse = data.get("sessioni_rimosse", 0)

                            # Mostra un messaggio di successo più dettagliato
                            st.success(
                                f"Reset completato! Rimosse {risposte_rimosse} risposte e {sessioni_rimosse} sessioni di memoria."
                            )

                        else:
                            st.error(f"Errore API ({response.status_code})")
                    except Exception as e:
                        st.error(f"Errore di connessione: {e}")

            if st.button("Esporta JSON", width="stretch"):
                if st.session_state.messages:
                    json_data = export_conversation_json()
                    st.download_button(
                        "Download JSON",
                        json_data,
                        file_name=f"blueai_chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json",
                        width="stretch",
                    )
                else:
                    st.info("Nessuna conversazione da esportare")

            if st.button("Esporta TXT", width="stretch"):
                if st.session_state.messages:
                    txt_data = export_conversation_txt()
                    st.download_button(
                        "Download TXT",
                        txt_data,
                        file_name=f"blueai_chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                        mime="text/plain",
                        width="stretch",
                    )
                else:
                    st.info("Nessuna conversazione da esportare")

        with tab2:
            st.markdown("#### Query Preferite")

            # Aggiungi query predefinite se la lista è vuota
            if not st.session_state.favorite_queries:
                st.session_state.favorite_queries = [
                    "Mostrami tutti i clienti",
                    "Qual è il fatturato totale?",
                    "Lista delle ditte",
                ]

            for idx, fav_query in enumerate(st.session_state.favorite_queries):
                col1, col2 = st.columns([4, 1])
                with col1:
                    if st.button(f"{fav_query}", key=f"fav_{idx}", width="stretch"):
                        st.session_state.pending_query = fav_query
                        st.rerun()
                with col2:
                    if st.button("🗑️", key=f"del_fav_{idx}"):
                        st.session_state.favorite_queries.pop(idx)
                        st.rerun()

            # Aggiungi nuova query preferita
            with st.expander("Aggiungi Preferito"):
                new_fav = st.text_input("Nuova query", key="new_favorite")
                if st.button("Salva", key="save_fav"):
                    if new_fav and add_to_favorites(new_fav):
                        st.success(" Aggiunto!")
                        st.rerun()
                    elif new_fav:
                        st.warning("Già presente")

        with tab3:
            st.markdown("#### Statistiche Sessione")

            total = st.session_state.conversation_stats["total_queries"]
            successful = st.session_state.conversation_stats["successful_queries"]
            success_rate = (successful / total * 100) if total > 0 else 0

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Query Totali", total)
            with col2:
                st.metric("Successo", f"{success_rate:.0f}%")

        st.metric("Messaggi", len(st.session_state.messages))

        start_time = datetime.fromisoformat(
            st.session_state.conversation_stats["start_time"]
        )
        duration = datetime.now() - start_time
        minutes = int(duration.total_seconds() / 60)
        st.caption(f"⏱️ Sessione: {minutes} minuti")

        with st.expander("📈 Slow query log"):
            entries = load_slow_query_log()
            if entries:
                df_log = pd.DataFrame(entries)
                if "query_originale" in df_log.columns:
                    df_log["query_originale"] = df_log["query_originale"].apply(
                        lambda q: (
                            (q[:200] + "…")
                            if isinstance(q, str) and len(q) > 220
                            else q
                        )
                    )
                st.dataframe(df_log, use_container_width=True)
            else:
                st.caption("Nessuna query lenta registrata al momento.")

        st.markdown("---")

        # Query rapide
        st.markdown("### 🚀 Query Veloci")

        quick_queries = [
            ("Lista Clienti", "Mostrami tutti i clienti"),
            ("Fatturato Totale", "Qual è il fatturato totale?"),
            ("Lista Ditte", "Mostra tutte le ditte"),
            ("Top Cliente", "Chi è il cliente con più fatturato?"),
            ("Statistiche", "Dammi delle statistiche generali"),
            ("Analisi Rapida", "Fai un'analisi generale dei dati"),
        ]

        for icon_label, query in quick_queries:
            if st.button(icon_label, width="stretch", key=f"quick_{query}"):
                st.session_state.pending_query = query
                st.rerun()

        st.markdown("---")

        # Tips
        with st.expander("💡 Tips & Tricks"):
            st.markdown(
                """
            **Performance:**
            - Domande chiare = risposte veloci
            - Cache attiva per query ripetute
            - Esempi: *"clienti ditta X"*, *"fatturato Y"*
            
            **Funzionalità:**
            - Salva query preferite
            - Esporta conversazioni
            - Suggerimenti automatici
            - Statistiche in tempo reale
            
            **Shortcut:**
            - Query veloci nella sidebar
            - Click su suggerimenti correlati
            - Export JSON/TXT disponibile
            """
            )
else:
    # Modalità Cliente: tutto spento, tranne i suggerimenti
    debug_mode = False
    show_context = False
    show_query = False
    show_graph = True  # Puoi decidere se tenerlo True per i clienti
    show_table = True  # Puoi decidere se tenerlo True per i clienti
    show_suggestions = True  # Puoi decidere se tenerlo True per i clienti

# --- AREA PRINCIPALE CHAT ---

# Messaggio di benvenuto se chat vuota
if not st.session_state.messages:
    st.markdown(
        """
    <div style='text-align: center; padding: 3rem 2rem; background: linear-gradient(135deg, rgba(59, 130, 246, 0.1), rgba(147, 197, 253, 0.1)); border-radius: 20px; border: 1px solid rgba(59, 130, 246, 0.3); margin: 2rem 0;'>
        <h2 style='color: #3b82f6; margin-bottom: 1rem;'> Benvenuto in BlueAI</h2>
        <p style='color: #cbd5e1; font-size: 1.1rem; margin-bottom: 2rem;'>
            Il tuo assistente AI per interrogare il Knowledge Graph Neo4j.<br>
            Fai una domanda per iniziare!
        </p>
        <div style='display: flex; gap: 1rem; justify-content: center; flex-wrap: wrap;'>
            <span style='background: rgba(59, 130, 246, 0.2); padding: 0.5rem 1rem; border-radius: 20px; color: #93c5fd;'>
                 Linguaggio naturale
            </span>
            <span style='background: rgba(59, 130, 246, 0.2); padding: 0.5rem 1rem; border-radius: 20px; color: #93c5fd;'>
                 Query Cypher automatiche
            </span>
            <span style='background: rgba(59, 130, 246, 0.2); padding: 0.5rem 1rem; border-radius: 20px; color: #93c5fd;'>
                Analisi intelligenti
            </span>
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

# Mostra messaggi precedenti
for i, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        # Mostra query Cypher solo se presente e richiesto
        if (
            message["role"] == "assistant"
            and show_query
            and message.get("query")
            and message["query"] not in ["N/D", "Errore", ""]
        ):
            st.code(message["query"], language="cypher")

        # Mostra debug se richiesto
        if debug_mode and message["role"] == "assistant":
            debug_data = {
                "success": message.get("success", False),
                "has_query": bool(message.get("query")),
                "has_context": bool(message.get("context")),
                "timestamp": message.get("timestamp", "N/A"),
                "complexity": (message.get("timing_details") or {}).get(
                    "query_complexity"
                ),
                "safe_rewrite": (message.get("timing_details") or {}).get(
                    "safe_rewrite"
                ),
            }
            with st.expander("🔍 Debug Info"):
                st.json(debug_data)

        # Mostra context se richiesto e presente
        if show_context and message["role"] == "assistant" and message.get("context"):
            with st.expander("Dati Raw dal Grafo"):
                st.json(message["context"])

        if show_table and message["role"] == "assistant" and message.get("context"):
            with st.expander("Tabella risultati"):
                if not render_table(message["context"]):
                    st.info("Contenuto non tabellare")

        if show_graph and message["role"] == "assistant" and message.get("graph_data"):
            with st.expander("Visualizzazione Grafo"):
                render_graph(
                    message["graph_data"], element_id=f"neo4j-graph-history-{i}"
                )
        elif show_graph and message["role"] == "assistant":
            with st.expander("Visualizzazione Grafo"):
                origin = None
                if isinstance(message.get("timing_details"), dict):
                    origin = message["timing_details"].get("graph_origin")
                if origin == "reconstructed_from_query":
                    st.info(
                        "Grafo ricostruito ma vuoto (nessun nodo distinto trovato)."
                    )
                elif origin == "context_results":
                    st.info("La query non ha restituito nodi Neo4j da visualizzare.")
                else:
                    st.info("Nessun grafo disponibile per questa risposta.")

        if message["role"] == "assistant":
            render_execution_details_ui(
                message.get("timing_details"),
                label=f"Dettagli esecuzione • risposta #{i + 1}",
            )
            prev_question = None
            if i > 0:
                previous = st.session_state.messages[i - 1]
                if previous.get("role") == "user":
                    prev_question = previous.get("content")
            feedback_key = message.get("message_id") or f"history_{i}"
            show_feedback_form(
                user_question=prev_question,
                answer=message.get("content"),
                query=message.get("query"),
                timing_details=message.get("timing_details"),
                key=feedback_key,
            )

# Gestione query pendenti (da bottoni)
if "pending_query" in st.session_state:
    query = st.session_state.pending_query
    del st.session_state.pending_query
    process_query(query)
    st.rerun()

# Input chat
if prompt := st.chat_input("Chiedi qualsiasi cosa sul tuo Knowledge Graph..."):
    process_query(prompt)
    st.rerun()

# Footer fisso in basso
st.markdown(
    """
    <div class="footer-sticky">
        <span class="float-icon">💡</span> <strong>Tip:</strong> 
        Usa le query veloci o salva le tue preferite | 
        <strong>Powered by</strong> BlueSystem × BlueAI 
    </div>
    """,
    unsafe_allow_html=True,
)
