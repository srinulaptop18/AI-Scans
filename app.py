import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
import numpy as np
from datetime import datetime
import io
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image as RLImage
from reportlab.lib.enums import TA_CENTER, TA_LEFT
import torchvision.transforms as transforms
import timm
import base64
import os

# Page configuration
st.set_page_config(
    page_title="NeuroScan AI - Parkinson's MRI Analysis",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Define the ResNetViT model architecture (Original)
class ResNetViT(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = timm.create_model("resnet50", pretrained=True, num_classes=0)
        feat_dim = self.backbone.num_features
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feat_dim, nhead=8, dim_feedforward=2048, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.head = nn.Linear(feat_dim, num_classes)
    
    def forward(self, x):
        feats = self.backbone(x)
        feats = feats.unsqueeze(1)
        feats = self.transformer(feats)
        return self.head(feats[:, 0])

# Define the EfficientNet-MobileNet model architecture (Old)
class EfficientNetMobileNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.efficientnet = timm.create_model('efficientnet_b0', pretrained=True, num_classes=0)
        efficientnet_features = self.efficientnet.num_features
        self.mobilenet = timm.create_model('mobilenetv3_large_100', pretrained=True, num_classes=0)
        mobilenet_features = self.mobilenet.num_features
        combined_features = efficientnet_features + mobilenet_features
        self.classifier = nn.Sequential(
            nn.Linear(combined_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        eff_features = self.efficientnet(x)
        mob_features = self.mobilenet(x)
        combined = torch.cat([eff_features, mob_features], dim=1)
        return self.classifier(combined)

# Define EfficientNetV2Small model
class EfficientNetV2Small(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = timm.create_model('tf_efficientnetv2_s', pretrained=True, num_classes=num_classes)
    
    def forward(self, x):
        return self.model(x)

# Helper function to load and encode logo image as base64
def get_logo_base64(logo_path):
    try:
        if os.path.exists(logo_path):
            with open(logo_path, "rb") as f:
                data = base64.b64encode(f.read()).decode()
            ext = logo_path.split(".")[-1].lower()
            mime = "image/jpeg" if ext in ["jpg", "jpeg"] else f"image/{ext}"
            return f"data:{mime};base64,{data}"
    except:
        pass
    return None

# Premium CSS — Enhanced Edition
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@500;600;700&family=Exo+2:wght@300;400;600;700;800&family=Share+Tech+Mono&display=swap');

    /* ── CSS Variables ─────────────────────────────────── */
    :root {
        --cyan:      #00d4ff;
        --cyan-dim:  rgba(0,212,255,0.18);
        --cyan-glow: rgba(0,212,255,0.45);
        --blue:      #0052cc;
        --blue-mid:  #0099e6;
        --green:     #00e676;
        --red:       #ff4444;
        --bg-deep:   #020817;
        --bg-card:   #0a1628;
        --bg-dark:   #040e1c;
        --text-main: #e0f4ff;
        --text-mute: #7ecfee;
        --text-pale: #94d8ff;
    }

    /* ── Keyframes ─────────────────────────────────────── */
    @keyframes borderRun {
        0%   { background-position: 0% 50%; }
        100% { background-position: 200% 50%; }
    }
    @keyframes fadeUp {
        from { opacity: 0; transform: translateY(28px); }
        to   { opacity: 1; transform: translateY(0); }
    }
    @keyframes fadeIn {
        from { opacity: 0; }
        to   { opacity: 1; }
    }
    @keyframes pulse-ring {
        0%   { box-shadow: 0 0 0 0 var(--cyan-dim), 0 0 0 0 var(--cyan-dim); }
        70%  { box-shadow: 0 0 0 10px transparent, 0 0 0 20px transparent; }
        100% { box-shadow: 0 0 0 0 transparent; }
    }
    @keyframes scanline {
        0%   { transform: translateY(-100%); }
        100% { transform: translateY(100vh); }
    }
    @keyframes shimmer {
        0%   { background-position: -200% center; }
        100% { background-position: 200% center; }
    }
    @keyframes glitch {
        0%,100% { clip-path: inset(0 0 98% 0); transform: translate(0); }
        20%      { clip-path: inset(30% 0 50% 0); transform: translate(-3px, 2px); }
        40%      { clip-path: inset(60% 0 20% 0); transform: translate(3px, -2px); }
        60%      { clip-path: inset(10% 0 80% 0); transform: translate(-2px, 1px); }
        80%      { clip-path: inset(80% 0 5%  0); transform: translate(2px, -1px); }
    }
    @keyframes float {
        0%,100% { transform: translateY(0px); }
        50%      { transform: translateY(-6px); }
    }
    @keyframes rotateOrbit {
        from { transform: rotate(0deg) translateX(42px) rotate(0deg); }
        to   { transform: rotate(360deg) translateX(42px) rotate(-360deg); }
    }
    @keyframes bgPan {
        0%   { background-position: 0% 0%; }
        100% { background-position: 100% 100%; }
    }
    @keyframes neonFlicker {
        0%,19%,21%,23%,25%,54%,56%,100% { text-shadow: 0 0 10px var(--cyan), 0 0 30px var(--cyan), 0 0 60px var(--cyan-glow); }
        20%,24%,55% { text-shadow: none; }
    }
    @keyframes spin-slow {
        from { transform: rotate(0deg); }
        to   { transform: rotate(360deg); }
    }
    @keyframes dash {
        to { stroke-dashoffset: 0; }
    }

    /* ── Base ──────────────────────────────────────────── */
    html, body, .stApp {
        background: var(--bg-deep) !important;
        font-family: 'Exo 2', sans-serif !important;
    }

    /* Animated grid background overlay */
    .stApp::before {
        content: '';
        position: fixed;
        inset: 0;
        background-image:
            linear-gradient(rgba(0,212,255,0.03) 1px, transparent 1px),
            linear-gradient(90deg, rgba(0,212,255,0.03) 1px, transparent 1px);
        background-size: 60px 60px;
        pointer-events: none;
        z-index: 0;
        animation: bgPan 60s linear infinite;
    }

    /* Ambient radial glow behind content */
    .stApp::after {
        content: '';
        position: fixed;
        top: -20%;
        left: 50%;
        transform: translateX(-50%);
        width: 900px;
        height: 600px;
        background: radial-gradient(ellipse at center,
            rgba(0,82,204,0.12) 0%,
            rgba(0,212,255,0.06) 40%,
            transparent 70%);
        pointer-events: none;
        z-index: 0;
    }

    .block-container {
        padding-top: 1rem !important;
        max-width: 1300px !important;
        position: relative;
        z-index: 1;
    }
    #MainMenu, footer, header { visibility: hidden; }

    /* ── Typography ────────────────────────────────────── */
    h2 {
        font-family: 'Rajdhani', sans-serif !important;
        font-size: 1.5rem !important;
        font-weight: 700 !important;
        color: #ffffff !important;
        letter-spacing: 3px !important;
        border-left: 4px solid var(--cyan) !important;
        padding-left: 14px !important;
        margin: 1.8rem 0 1.1rem 0 !important;
        text-transform: uppercase !important;
        position: relative !important;
    }
    h2::after {
        content: '' !important;
        position: absolute !important;
        bottom: -6px !important;
        left: 18px !important;
        width: 40px !important;
        height: 2px !important;
        background: var(--cyan) !important;
        border-radius: 2px !important;
    }
    h3 {
        font-family: 'Rajdhani', sans-serif !important;
        font-size: 1.15rem !important;
        font-weight: 600 !important;
        color: var(--text-pale) !important;
        letter-spacing: 1.5px !important;
        margin: 1.1rem 0 0.6rem 0 !important;
    }

    /* ── Neo Cards ─────────────────────────────────────── */
    .neo-card {
        background: linear-gradient(145deg, #0d1b2e 0%, #0a1628 60%, #071220 100%);
        border: 1px solid rgba(0, 212, 255, 0.22);
        border-radius: 20px;
        padding: 2rem;
        margin: 1rem 0;
        box-shadow:
            0 4px 40px rgba(0,0,0,0.7),
            0 0 80px rgba(0,82,204,0.08),
            inset 0 1px 0 rgba(255,255,255,0.06),
            inset 0 -1px 0 rgba(0,0,0,0.4);
        position: relative;
        overflow: hidden;
        animation: fadeUp 0.55s cubic-bezier(0.22,1,0.36,1);
        transition: border-color 0.3s, box-shadow 0.3s;
    }
    .neo-card:hover {
        border-color: rgba(0,212,255,0.45);
        box-shadow:
            0 8px 60px rgba(0,0,0,0.7),
            0 0 100px rgba(0,82,204,0.14),
            inset 0 1px 0 rgba(255,255,255,0.08);
    }
    /* Top cyan shimmer line */
    .neo-card::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 1px;
        background: linear-gradient(90deg,
            transparent 0%,
            rgba(0,212,255,0.8) 40%,
            rgba(0,212,255,1)   50%,
            rgba(0,212,255,0.8) 60%,
            transparent 100%);
    }
    /* Corner accent — top-left */
    .neo-card::after {
        content: '';
        position: absolute;
        top: 0; left: 0;
        width: 40px; height: 40px;
        border-top: 2px solid rgba(0,212,255,0.5);
        border-left: 2px solid rgba(0,212,255,0.5);
        border-radius: 20px 0 0 0;
    }

    /* ── Form Inputs ───────────────────────────────────── */
    .stTextInput input, .stNumberInput input,
    .stTextArea textarea, .stDateInput input {
        background: linear-gradient(145deg, #04101e, #061020) !important;
        color: var(--text-main) !important;
        border: 1px solid rgba(0,180,220,0.3) !important;
        border-radius: 10px !important;
        font-family: 'Exo 2', sans-serif !important;
        font-size: 0.95rem !important;
        transition: border-color 0.25s, box-shadow 0.25s, background 0.25s !important;
        padding: 0.55rem 0.9rem !important;
    }
    .stTextInput input:hover, .stNumberInput input:hover,
    .stTextArea textarea:hover {
        border-color: rgba(0,212,255,0.5) !important;
        background: linear-gradient(145deg, #051320, #081828) !important;
    }
    .stTextInput input:focus, .stNumberInput input:focus,
    .stTextArea textarea:focus, .stDateInput input:focus {
        border-color: var(--cyan) !important;
        box-shadow: 0 0 0 3px rgba(0,212,255,0.14), 0 0 20px rgba(0,212,255,0.08) !important;
        background: linear-gradient(145deg, #061526, #081e30) !important;
    }
    div[data-baseweb="select"] > div {
        background: linear-gradient(145deg, #04101e, #061020) !important;
        border: 1px solid rgba(0,180,220,0.3) !important;
        border-radius: 10px !important;
        color: var(--text-main) !important;
        transition: border-color 0.25s !important;
    }
    div[data-baseweb="select"] > div:hover {
        border-color: rgba(0,212,255,0.5) !important;
    }
    label, .stSelectbox label, .stTextInput label,
    .stNumberInput label, .stDateInput label, .stTextArea label {
        color: var(--text-mute) !important;
        font-family: 'Exo 2', sans-serif !important;
        font-size: 0.82rem !important;
        font-weight: 700 !important;
        letter-spacing: 1.2px !important;
        text-transform: uppercase !important;
    }

    /* ── Buttons ───────────────────────────────────────── */
    .stButton > button {
        background: linear-gradient(135deg, #003fa3 0%, #0066cc 50%, #0099e6 100%) !important;
        color: #ffffff !important;
        border: 1px solid rgba(0,180,255,0.4) !important;
        border-radius: 12px !important;
        padding: 0.85rem 1.5rem !important;
        font-family: 'Rajdhani', sans-serif !important;
        font-size: 1.15rem !important;
        font-weight: 700 !important;
        letter-spacing: 3px !important;
        text-transform: uppercase !important;
        width: 100% !important;
        transition: all 0.3s cubic-bezier(0.34,1.56,0.64,1) !important;
        box-shadow:
            0 4px 20px rgba(0,100,200,0.5),
            0 0 0 0 rgba(0,212,255,0),
            inset 0 1px 0 rgba(255,255,255,0.15) !important;
        position: relative !important;
        overflow: hidden !important;
    }
    .stButton > button::before {
        content: '' !important;
        position: absolute !important;
        top: 0 !important; left: -100% !important;
        width: 100% !important; height: 100% !important;
        background: linear-gradient(90deg,
            transparent,
            rgba(255,255,255,0.12),
            transparent) !important;
        transition: left 0.5s ease !important;
    }
    .stButton > button:hover::before { left: 100% !important; }
    .stButton > button:hover {
        background: linear-gradient(135deg, #0052cc 0%, #0088dd 50%, #00c0ff 100%) !important;
        transform: translateY(-3px) scale(1.01) !important;
        box-shadow:
            0 10px 32px rgba(0,140,255,0.6),
            0 0 40px rgba(0,212,255,0.2),
            inset 0 1px 0 rgba(255,255,255,0.2) !important;
        border-color: rgba(0,212,255,0.7) !important;
    }
    .stButton > button:active {
        transform: translateY(-1px) scale(0.99) !important;
        box-shadow: 0 4px 16px rgba(0,100,200,0.4) !important;
    }

    .stDownloadButton > button {
        background: linear-gradient(135deg, #004d25 0%, #007a40 50%, #00aa55 100%) !important;
        color: #fff !important;
        border: 1px solid rgba(0,200,100,0.35) !important;
        border-radius: 12px !important;
        padding: 0.85rem 1.5rem !important;
        font-family: 'Rajdhani', sans-serif !important;
        font-size: 1.15rem !important;
        font-weight: 700 !important;
        letter-spacing: 3px !important;
        text-transform: uppercase !important;
        width: 100% !important;
        transition: all 0.3s cubic-bezier(0.34,1.56,0.64,1) !important;
        box-shadow: 0 4px 20px rgba(0,160,80,0.45), inset 0 1px 0 rgba(255,255,255,0.12) !important;
        position: relative !important;
        overflow: hidden !important;
    }
    .stDownloadButton > button:hover {
        transform: translateY(-3px) scale(1.01) !important;
        box-shadow: 0 10px 30px rgba(0,200,100,0.55), 0 0 40px rgba(0,230,118,0.18) !important;
        border-color: rgba(0,230,118,0.6) !important;
    }

    /* ── File Uploader ─────────────────────────────────── */
    [data-testid="stFileUploader"] {
        background: linear-gradient(145deg, #060f1c, #091522) !important;
        border: 2px dashed rgba(0,212,255,0.35) !important;
        border-radius: 18px !important;
        padding: 1.8rem !important;
        transition: all 0.35s ease !important;
        position: relative !important;
    }
    [data-testid="stFileUploader"]:hover {
        border-color: var(--cyan) !important;
        box-shadow:
            0 0 40px rgba(0,212,255,0.12),
            inset 0 0 30px rgba(0,212,255,0.05) !important;
        background: linear-gradient(145deg, #071424, #0c1e30) !important;
    }

    /* ── Result Cards ──────────────────────────────────── */
    .result-card {
        background: linear-gradient(145deg, #071424, #0c1f38, #081830);
        border: 1px solid rgba(0,212,255,0.28);
        border-radius: 18px;
        padding: 1.6rem 2rem;
        margin: 0.5rem 0;
        text-align: center;
        position: relative;
        overflow: hidden;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        animation: fadeUp 0.6s cubic-bezier(0.22,1,0.36,1);
    }
    .result-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 12px 40px rgba(0,0,0,0.5), 0 0 30px rgba(0,212,255,0.12);
    }
    /* Top glint */
    .result-card::before {
        content: '';
        position: absolute;
        top: 0; left: 20%; right: 20%;
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(0,212,255,0.6), transparent);
    }
    /* Animated bottom bar */
    .result-card::after {
        content: '';
        position: absolute;
        bottom: 0; left: 0; right: 0;
        height: 3px;
        background: linear-gradient(90deg, #0052cc, #00d4ff, #00ff88, #00d4ff, #0052cc);
        background-size: 300% 100%;
        animation: borderRun 2.5s linear infinite;
    }
    .result-label {
        font-family: 'Share Tech Mono', monospace;
        font-size: 0.78rem;
        color: var(--text-mute);
        letter-spacing: 3px;
        text-transform: uppercase;
        margin-bottom: 0.6rem;
        opacity: 0.9;
    }
    .result-value-normal {
        font-family: 'Rajdhani', sans-serif;
        font-size: 2.6rem;
        font-weight: 700;
        color: var(--green);
        letter-spacing: 3px;
        text-shadow:
            0 0 20px rgba(0,230,118,0.7),
            0 0 60px rgba(0,230,118,0.25);
    }
    .result-value-parkinson {
        font-family: 'Rajdhani', sans-serif;
        font-size: 2.6rem;
        font-weight: 700;
        color: var(--red);
        letter-spacing: 3px;
        text-shadow:
            0 0 20px rgba(255,68,68,0.7),
            0 0 60px rgba(255,68,68,0.25);
    }
    .result-value-confidence {
        font-family: 'Rajdhani', sans-serif;
        font-size: 2.6rem;
        font-weight: 700;
        color: var(--cyan);
        letter-spacing: 3px;
        text-shadow:
            0 0 20px var(--cyan-glow),
            0 0 60px rgba(0,212,255,0.2);
        animation: neonFlicker 8s infinite;
    }

    /* ── MRI Image Frame ───────────────────────────────── */
    .img-frame {
        border: 1px solid rgba(0,212,255,0.3);
        border-radius: 18px;
        padding: 0.85rem;
        background: var(--bg-dark);
        box-shadow:
            0 0 40px rgba(0,212,255,0.1),
            inset 0 0 20px rgba(0,0,0,0.6);
        position: relative;
        transition: box-shadow 0.3s;
    }
    .img-frame:hover {
        box-shadow:
            0 0 60px rgba(0,212,255,0.18),
            inset 0 0 20px rgba(0,0,0,0.6);
    }
    /* Corner bracket decorations */
    .img-frame::before {
        content: '';
        position: absolute;
        top: 6px; left: 6px;
        width: 22px; height: 22px;
        border-top: 2px solid var(--cyan);
        border-left: 2px solid var(--cyan);
        border-radius: 4px 0 0 0;
    }
    .img-frame::after {
        content: '';
        position: absolute;
        bottom: 6px; right: 6px;
        width: 22px; height: 22px;
        border-bottom: 2px solid var(--cyan);
        border-right: 2px solid var(--cyan);
        border-radius: 0 0 4px 0;
    }

    /* ── Metric Widgets ────────────────────────────────── */
    [data-testid="stMetric"] {
        background: linear-gradient(145deg, #071424, #0c1f38) !important;
        border: 1px solid rgba(0,212,255,0.18) !important;
        border-radius: 16px !important;
        padding: 1.1rem 1.5rem !important;
        transition: transform 0.25s, box-shadow 0.25s !important;
    }
    [data-testid="stMetric"]:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 30px rgba(0,0,0,0.4), 0 0 20px rgba(0,212,255,0.1) !important;
    }
    [data-testid="stMetricValue"] {
        font-family: 'Rajdhani', sans-serif !important;
        font-size: 1.9rem !important;
        font-weight: 700 !important;
        color: var(--cyan) !important;
        text-shadow: 0 0 12px rgba(0,212,255,0.4) !important;
    }
    [data-testid="stMetricLabel"] {
        color: var(--text-mute) !important;
        font-family: 'Share Tech Mono', monospace !important;
        font-size: 0.75rem !important;
        font-weight: 400 !important;
        letter-spacing: 1.5px !important;
        text-transform: uppercase !important;
    }

    /* ── Sidebar ───────────────────────────────────────── */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #020c18 0%, #030e1e 100%) !important;
        border-right: 1px solid rgba(0,212,255,0.12) !important;
        box-shadow: 4px 0 30px rgba(0,0,0,0.5) !important;
    }
    [data-testid="stSidebar"]::before {
        content: '';
        position: absolute;
        top: 0; bottom: 0; right: 0;
        width: 1px;
        background: linear-gradient(180deg,
            transparent 0%,
            rgba(0,212,255,0.3) 30%,
            rgba(0,212,255,0.5) 50%,
            rgba(0,212,255,0.3) 70%,
            transparent 100%);
    }
    [data-testid="stSidebar"] * { font-family: 'Exo 2', sans-serif !important; }

    /* ── Tabs ──────────────────────────────────────────── */
    .stTabs [data-baseweb="tab-list"] {
        background: linear-gradient(135deg, #040e1c, #061424) !important;
        border: 1px solid rgba(0,212,255,0.15) !important;
        border-radius: 14px !important;
        padding: 0.4rem !important;
        gap: 0.3rem !important;
        box-shadow: 0 4px 20px rgba(0,0,0,0.4) !important;
    }
    .stTabs [data-baseweb="tab"] {
        font-family: 'Rajdhani', sans-serif !important;
        font-size: 1rem !important;
        font-weight: 700 !important;
        letter-spacing: 2px !important;
        text-transform: uppercase !important;
        color: var(--text-mute) !important;
        border-radius: 10px !important;
        transition: all 0.25s ease !important;
        padding: 0.6rem 1.4rem !important;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: var(--cyan) !important;
        background: rgba(0,212,255,0.06) !important;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, rgba(0,82,204,0.4), rgba(0,212,255,0.18)) !important;
        color: #ffffff !important;
        box-shadow: 0 0 20px rgba(0,212,255,0.15) !important;
        border: 1px solid rgba(0,212,255,0.3) !important;
    }
    .stTabs [data-baseweb="tab-highlight"] {
        background: none !important;
    }
    .stTabs [data-baseweb="tab-border"] {
        display: none !important;
    }

    /* ── Alert / Info boxes ────────────────────────────── */
    .stAlert {
        border-radius: 14px !important;
        border-left-width: 4px !important;
        backdrop-filter: blur(4px) !important;
    }
    .stInfo {
        background: rgba(0,82,204,0.12) !important;
        border-color: var(--blue-mid) !important;
    }
    .stSuccess {
        background: rgba(0,160,80,0.1) !important;
    }
    .stWarning {
        background: rgba(200,120,0,0.1) !important;
    }

    /* ── Spinner ───────────────────────────────────────── */
    .stSpinner > div {
        border-color: var(--cyan) transparent transparent transparent !important;
    }

    /* ── HR Divider ────────────────────────────────────── */
    hr {
        border: none !important;
        height: 1px !important;
        background: linear-gradient(90deg,
            transparent 0%,
            rgba(0,82,204,0.4) 20%,
            rgba(0,212,255,0.7) 50%,
            rgba(0,82,204,0.4) 80%,
            transparent 100%) !important;
        margin: 2.2rem 0 !important;
        position: relative !important;
    }

    /* ── Scrollbar ─────────────────────────────────────── */
    ::-webkit-scrollbar { width: 5px; height: 5px; }
    ::-webkit-scrollbar-track {
        background: #020817;
        border-radius: 10px;
    }
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(180deg, #0052cc, #00d4ff);
        border-radius: 10px;
        box-shadow: 0 0 6px rgba(0,212,255,0.4);
    }
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(180deg, #0066ff, #00eeff);
    }

    /* ── Selection ─────────────────────────────────────── */
    ::selection {
        background: rgba(0,212,255,0.25);
        color: #ffffff;
    }

    /* ── Tooltip / Popover ─────────────────────────────── */
    [data-baseweb="popover"] {
        background: #061424 !important;
        border: 1px solid rgba(0,212,255,0.25) !important;
        border-radius: 12px !important;
        box-shadow: 0 8px 40px rgba(0,0,0,0.7) !important;
    }

    /* ── Progress / Balloons ───────────────────────────── */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, var(--blue), var(--cyan)) !important;
        border-radius: 10px !important;
        box-shadow: 0 0 10px rgba(0,212,255,0.4) !important;
    }
</style>
""", unsafe_allow_html=True)

# ── HERO HEADER ──────────────────────────────────────────────────────────────
# NeuroScan AI logo = logo.png
logo_src = get_logo_base64("logo.png")

if logo_src:
    logo_html = (
        f'<img src="{logo_src}" style="width:90px;height:90px;object-fit:contain;'
        f'border-radius:50%;border:2px solid rgba(0,212,255,0.5);'
        f'box-shadow:0 0 30px rgba(0,212,255,0.3),inset 0 0 20px rgba(0,212,255,0.1);'
        f'margin:0 auto 1.2rem;display:block;" />'
    )
else:
    logo_html = (
        '<div style="width:90px;height:90px;background:radial-gradient(circle,#0a2040 0%,#020c18 70%);'
        'border:2px solid rgba(0,212,255,0.5);border-radius:50%;display:flex;align-items:center;'
        'justify-content:center;margin:0 auto 1.2rem;font-size:2.8rem;'
        'box-shadow:0 0 30px rgba(0,212,255,0.3),inset 0 0 20px rgba(0,212,255,0.1);">&#x1F9E0;</div>'
    )

header_html = (
    '<div style="background:linear-gradient(180deg,#020c18 0%,#041428 100%);border-bottom:1px solid rgba(0,212,255,0.25);padding:2.5rem 2rem 2rem;text-align:center;position:relative;overflow:hidden;">'
    '<div style="width:80px;height:3px;margin:0 auto 1.4rem;background:linear-gradient(90deg,#0052cc,#00d4ff);border-radius:2px;"></div>'
    + logo_html +
    '<div style="font-family:Rajdhani,sans-serif;font-size:3.8rem;font-weight:700;color:#ffffff;letter-spacing:8px;text-transform:uppercase;line-height:1;margin-bottom:0.6rem;text-shadow:0 0 40px rgba(0,180,255,0.4);">NEUROSCAN <span style="color:#00d4ff;">AI</span></div>'
    '<div style="font-family:\'Exo 2\',sans-serif;font-size:1rem;color:#7ecfee;letter-spacing:4px;text-transform:uppercase;margin-bottom:1.6rem;">Parkinsons Disease &nbsp;&middot;&nbsp; MRI Analysis &nbsp;&middot;&nbsp; Deep Learning</div>'
    '<div style="display:inline-flex;gap:2rem;background:rgba(0,212,255,0.06);border:1px solid rgba(0,212,255,0.18);border-radius:50px;padding:0.6rem 2rem;font-size:0.82rem;letter-spacing:1px;color:#94d8ff;">'
    '<span>&#129302; ResNet-ViT </span>'
    '<span style="color:rgba(0,212,255,0.3);">|</span>'
    '<span>&#127919; 99.4% Accuracy</span>'
    '<span style="color:rgba(0,212,255,0.3);">|</span>'
    '<span>&#9889; Real-time Analysis</span>'
    '</div>'
    '<div style="width:80px;height:3px;margin:1.4rem auto 0;background:linear-gradient(90deg,#00d4ff,#0052cc);border-radius:2px;"></div>'
    '</div>'
)
st.markdown(header_html, unsafe_allow_html=True)

# Initialize session state
if 'prediction_made' not in st.session_state:
    st.session_state.prediction_made = False
if 'patient_data' not in st.session_state:
    st.session_state.patient_data = {}
if 'prediction_result' not in st.session_state:
    st.session_state.prediction_result = {}

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🔬 About NeuroScan AI")
    st.info("""
    **NeuroScan AI** is a cutting-edge deep learning system for analyzing brain MRI scans.
    
    **🎯 Features:**
    - 🧠 Advanced AI Analysis
    - 📊 Real-time Predictions  
    - 📄 Professional Reports
    - ⚡ Fast Processing
    """)
    st.markdown("---")
    st.markdown("### 🤖 Model Specs")
    st.success("""
    **Supported Models:**
    - ResNet50 + Transformer (ViT)
    
    **Classes:** Normal / Parkinson's
    **Accuracy:** ~99%
    **Status:** 🟢 Online
    """)
    st.markdown("---")
    st.markdown("### ⚠️ Medical Disclaimer")
    st.warning("⚕️ For research purposes only. Consult qualified medical professionals for clinical diagnosis.")
    st.markdown("---")
    st.markdown("### 📈 Statistics")
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        st.metric("Precision", "100%")
    with col_s2:
        st.metric("Recall", "98%")

# ── MAIN TABS ─────────────────────────────────────────────────────────────────
tab_scan, tab_about = st.tabs(["🧠  MRI Analysis", "🏫  About Us"])

# ════════════════════════════════════════════════════════════════════
#  TAB 1 — MRI ANALYSIS
# ════════════════════════════════════════════════════════════════════
with tab_scan:
    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.markdown("## 👤 Patient Information")
        st.markdown("<div class='neo-card'>", unsafe_allow_html=True)
        patient_name = st.text_input("📝 Full Name", placeholder="Enter patient's full name")
        col_age, col_gender = st.columns(2)
        with col_age:
            patient_age = st.number_input("🎂 Age", min_value=0, max_value=120, value=30)
        with col_gender:
            patient_gender = st.selectbox("⚧ Gender", ["Male", "Female", "Other"])
        patient_id = st.text_input("🆔 Patient ID", placeholder="e.g., P-2024-0001")
        scan_date = st.date_input("📅 Scan Date", value=datetime.now())
        medical_history = st.text_area("📋 Medical History (Optional)",
                                        placeholder="Brief medical history, symptoms, or relevant information...",
                                        height=100)
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown("## 🖼️ MRI Image Upload")
        st.markdown("<div class='neo-card'>", unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "🔬 Upload Brain MRI Scan",
            type=["png", "jpg", "jpeg"],
            help="Supported formats: PNG, JPG, JPEG"
        )
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.markdown("<div class='img-frame'>", unsafe_allow_html=True)
            st.image(image, caption="📸 Uploaded MRI Scan", use_column_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.info("⬆️ Please upload a brain MRI scan to begin analysis")
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("## 🔍 AI Analysis & Prediction")
    col_analyze, col_report = st.columns([1, 1], gap="large")

    with col_analyze:
        if st.button("🚀 ANALYZE MRI SCAN"):
            if not patient_name or not patient_id or uploaded_file is None:
                st.error("⚠️ Please fill in all required fields and upload an MRI scan")
            else:
                with st.spinner("🧠 Analyzing MRI scan with AI model..."):
                    st.session_state.patient_data = {
                        "name": patient_name,
                        "age": patient_age,
                        "gender": patient_gender,
                        "patient_id": patient_id,
                        "scan_date": scan_date.strftime("%Y-%m-%d"),
                        "medical_history": medical_history
                    }
                    try:
                        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                        model = None
                        model_loaded = False
                        model_type = None
                        loaded_file = None
                        
                        for filename in ["new.pth", "old.pth", "parkinsons_resnet_vit_latest.pth"]:
                            try:
                                checkpoint = torch.load(filename, map_location=device)
                                keys = list(checkpoint.keys())
                                
                                if any('blocks' in k for k in keys) and any('conv_stem' in k for k in keys):
                                    st.info(f"📂 Loading **EfficientNet-V2 Small** from `{filename}`")
                                    model = EfficientNetV2Small(num_classes=2)
                                    model.load_state_dict(checkpoint)
                                    model_type = "EfficientNet-V2 Small"
                                    loaded_file = filename
                                    model_loaded = True
                                    break
                                elif any("efficientnet" in k for k in keys) or any("mobilenet" in k for k in keys):
                                    st.info(f"📂 Loading EfficientNet-MobileNet from **{filename}**")
                                    model = EfficientNetMobileNet(num_classes=2)
                                    model.load_state_dict(checkpoint)
                                    model_type = "EfficientNet-MobileNet"
                                    loaded_file = filename
                                    model_loaded = True
                                    break
                                elif any("backbone" in k for k in keys) and any("transformer" in k for k in keys):
                                    st.info(f"📂 Loading ResNet-ViT from **{filename}**")
                                    model = ResNetViT(num_classes=2)
                                    model.load_state_dict(checkpoint)
                                    model_type = "ResNet-ViT"
                                    loaded_file = filename
                                    model_loaded = True
                                    break
                                else:
                                    for ModelClass, mtype in [
                                        (EfficientNetV2Small, "EfficientNet-V2 Small"),
                                        (EfficientNetMobileNet, "EfficientNet-MobileNet"),
                                        (ResNetViT, "ResNet-ViT"),
                                    ]:
                                        try:
                                            model = ModelClass(num_classes=2)
                                            model.load_state_dict(checkpoint)
                                            model_type = mtype
                                            loaded_file = filename
                                            model_loaded = True
                                            break
                                        except:
                                            pass
                                    if model_loaded:
                                        break
                            except FileNotFoundError:
                                continue
                            except Exception as e:
                                st.warning(f"⚠️ Could not load {filename}: {str(e)[:100]}")
                                continue

                        if not model_loaded:
                            st.error("❌ Model file not found!")
                            st.info("💡 Please add **new.pth** file to this folder")
                            st.stop()

                        st.success(f"✅ Loaded **{model_type}** from `{loaded_file}`")
                        model.to(device)
                        model.eval()

                        transform = transforms.Compose([
                            transforms.Resize((224, 224)),
                            transforms.ToTensor(),
                        ])
                        image_rgb = image.convert("RGB") if image.mode != "RGB" else image
                        img_tensor = transform(image_rgb).unsqueeze(0).to(device)

                        with torch.no_grad():
                            outputs = model(img_tensor)
                            probabilities = torch.nn.functional.softmax(outputs, dim=1)
                            confidence, predicted = torch.max(probabilities, 1)

                        class_names = ["Normal", "Parkinson's Disease"]
                        st.session_state.prediction_result = {
                            "prediction": class_names[predicted.item()],
                            "confidence": float(confidence.item() * 100),
                            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "image": image,
                            "probabilities": {
                                "Normal": float(probabilities[0][0].item() * 100),
                                "Parkinson's": float(probabilities[0][1].item() * 100),
                            }
                        }
                        st.session_state.prediction_made = True
                        st.success("✅ Analysis complete!")
                        st.balloons()
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Error during analysis: {str(e)}")

    if st.session_state.prediction_made:
        pred = st.session_state.prediction_result["prediction"]
        conf = st.session_state.prediction_result["confidence"]
        prob_data = st.session_state.prediction_result["probabilities"]
        is_normal = pred == "Normal"
        val_class = "result-value-normal" if is_normal else "result-value-parkinson"

        st.markdown("<div class='neo-card'>", unsafe_allow_html=True)
        st.markdown("### 📊 Diagnostic Results")
        col_pred, col_conf = st.columns(2)
        with col_pred:
            st.markdown(f"<div class='result-card'><div class='result-label'>🎯 Diagnosis</div><div class='{val_class}'>{pred}</div></div>", unsafe_allow_html=True)
        with col_conf:
            st.markdown(f"<div class='result-card'><div class='result-label'>💯 Confidence</div><div class='result-value-confidence'>{conf:.2f}%</div></div>", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        col_p1, col_p2 = st.columns(2)
        with col_p1:
            st.metric("✅ Normal", f"{prob_data['Normal']:.2f}%")
        with col_p2:
            parkinsons_prob = prob_data["Parkinson's"]
            st.metric("⚠️ Parkinson's", f"{parkinsons_prob:.2f}%")
        st.markdown(f"<p style='color:#7ecfee;font-size:0.85rem;margin-top:1rem;'>⏰ {st.session_state.prediction_result['timestamp']}</p>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_report:
        if st.session_state.prediction_made:
            if st.button("📄 GENERATE PDF REPORT"):
                with st.spinner("📝 Generating PDF report..."):
                    try:
                        buffer = io.BytesIO()
                        doc = SimpleDocTemplate(buffer, pagesize=letter)
                        story = []
                        styles = getSampleStyleSheet()
                        title_style = ParagraphStyle("T", parent=styles["Heading1"], fontSize=22,
                            textColor=colors.HexColor("#1565c0"), spaceAfter=24, alignment=TA_CENTER, fontName="Helvetica-Bold")
                        heading_style = ParagraphStyle("H", parent=styles["Heading2"], fontSize=15,
                            textColor=colors.HexColor("#1976d2"), spaceAfter=10, spaceBefore=10, fontName="Helvetica-Bold")
                        story.append(Paragraph("NEUROSCAN AI — PARKINSON'S DISEASE MRI REPORT", title_style))
                        story.append(Spacer(1, 16))
                        story.append(Paragraph("Patient Information", heading_style))
                        p = st.session_state.patient_data
                        patient_table = Table([
                            ["Field", "Details"],
                            ["Patient Name", p["name"]],
                            ["Patient ID", p["patient_id"]],
                            ["Age", str(p["age"])],
                            ["Gender", p["gender"]],
                            ["Scan Date", p["scan_date"]],
                        ], colWidths=[2*inch, 4*inch])
                        patient_table.setStyle(TableStyle([
                            ("BACKGROUND", (0,0),(-1,0), colors.HexColor("#1565c0")),
                            ("TEXTCOLOR", (0,0),(-1,0), colors.whitesmoke),
                            ("FONTNAME", (0,0),(-1,0), "Helvetica-Bold"),
                            ("FONTSIZE", (0,0),(-1,0), 12),
                            ("BOTTOMPADDING", (0,0),(-1,0), 10),
                            ("ROWBACKGROUNDS", (0,1),(-1,-1), [colors.white, colors.lightgrey]),
                            ("GRID", (0,0),(-1,-1), 1, colors.black),
                            ("FONTNAME", (0,1),(-1,-1), "Helvetica"),
                            ("FONTSIZE", (0,1),(-1,-1), 10),
                        ]))
                        story.append(patient_table)
                        story.append(Spacer(1, 16))
                        if p["medical_history"]:
                            story.append(Paragraph("Medical History", heading_style))
                            story.append(Paragraph(p["medical_history"], styles["Normal"]))
                            story.append(Spacer(1, 16))
                        story.append(Paragraph("AI Analysis Results", heading_style))
                        normal_prob = st.session_state.prediction_result["probabilities"]["Normal"]
                        parkinsons_prob = st.session_state.prediction_result["probabilities"]["Parkinson's"]
                        results_table = Table([
                            ["Metric", "Value"],
                            ["Diagnosis", st.session_state.prediction_result["prediction"]],
                            ["Confidence", f"{st.session_state.prediction_result['confidence']:.2f}%"],
                            ["Normal Probability", f"{normal_prob:.2f}%"],
                            ["Parkinson's Probability", f"{parkinsons_prob:.2f}%"],
                            ["Analysis Time", st.session_state.prediction_result["timestamp"]],
                            ["AI Model", "ResNet-ViT"],
                        ], colWidths=[2.5*inch, 3.5*inch])
                        results_table.setStyle(TableStyle([
                            ("BACKGROUND", (0,0),(-1,0), colors.HexColor("#1565c0")),
                            ("TEXTCOLOR", (0,0),(-1,0), colors.whitesmoke),
                            ("FONTNAME", (0,0),(-1,0), "Helvetica-Bold"),
                            ("FONTSIZE", (0,0),(-1,0), 12),
                            ("BOTTOMPADDING", (0,0),(-1,0), 10),
                            ("ROWBACKGROUNDS", (0,1),(-1,-1), [colors.white, colors.lightgrey]),
                            ("GRID", (0,0),(-1,-1), 1, colors.black),
                            ("FONTNAME", (0,1),(-1,-1), "Helvetica"),
                            ("FONTSIZE", (0,1),(-1,-1), 10),
                        ]))
                        story.append(results_table)
                        story.append(Spacer(1, 16))
                        story.append(Paragraph("Brain MRI Scan", heading_style))
                        img_buf = io.BytesIO()
                        st.session_state.prediction_result["image"].save(img_buf, format="PNG")
                        img_buf.seek(0)
                        story.append(RLImage(img_buf, width=4*inch, height=4*inch))
                        story.append(Spacer(1, 16))
                        story.append(Paragraph("Disclaimer", heading_style))
                        story.append(Paragraph(
                            "This report is generated by an AI system for research/educational purposes only. "
                            "It must not replace clinical diagnosis by qualified medical professionals.",
                            styles["Normal"]))
                        story.append(Spacer(1, 24))
                        footer_s = ParagraphStyle("F", parent=styles["Normal"], fontSize=8,
                            textColor=colors.grey, alignment=TA_CENTER)
                        story.append(Paragraph(f"NeuroScan AI — BVC College of Engineering, Palacharla — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", footer_s))
                        doc.build(story)
                        buffer.seek(0)
                        st.download_button(
                            label="⬇️ DOWNLOAD PDF REPORT",
                            data=buffer,
                            file_name=f"Parkinsons_MRI_Report_{p['patient_id']}_{datetime.now().strftime('%Y%m%d')}.pdf",
                            mime="application/pdf"
                        )
                        st.success("✅ PDF Report generated!")
                    except Exception as e:
                        st.error(f"❌ Error generating PDF: {str(e)}")
        else:
            st.info("📋 Complete the analysis first to generate a report")

# ════════════════════════════════════════════════════════════════════
#  TAB 2 — ABOUT US
# ════════════════════════════════════════════════════════════════════
with tab_about:
    # College logo = bvcr.jpg
    college_logo_src = get_logo_base64("bvcr.jpg")

    if college_logo_src:
        about_logo_html = (
            f'<img src="{college_logo_src}" style="width:110px;height:110px;object-fit:contain;'
            f'margin-bottom:1rem;filter:drop-shadow(0 0 12px rgba(6,182,212,0.5));" />'
        )
    else:
        about_logo_html = '<div style="font-size:3.5rem;margin-bottom:1rem;">🏛️</div>'

    st.markdown(f"""
<div style="background:linear-gradient(135deg,#f0f9ff,#e0f2fe);border:2px solid #06b6d4;border-radius:24px;padding:3rem;margin-bottom:2rem;text-align:center;box-shadow:0 0 30px rgba(6,182,212,0.3);">
    {about_logo_html}
    <div style="font-family:'Rajdhani',sans-serif;font-size:2.6rem;font-weight:700;color:#0c4a6e;letter-spacing:3px;margin-bottom:0.5rem;text-shadow:0 0 10px rgba(6,182,212,0.3);">BVC COLLEGE OF ENGINEERING</div>
    <div style="font-family:'Exo 2',sans-serif;font-size:1.2rem;color:#0891b2;letter-spacing:4px;text-transform:uppercase;margin-bottom:0.8rem;font-weight:600;">AUTONOMOUS</div>
    <div style="font-family:'Exo 2',sans-serif;font-size:1.1rem;color:#0369a1;letter-spacing:3px;text-transform:uppercase;margin-bottom:0.5rem;">Palacharla, Andhra Pradesh</div>
    <div style="width:100px;height:3px;background:linear-gradient(90deg,#06b6d4,#0891b2,#06b6d4);margin:1.2rem auto;border-radius:2px;box-shadow:0 0 10px rgba(6,182,212,0.5);"></div>
    <div style="font-family:'Exo 2',sans-serif;font-size:0.95rem;color:#0c4a6e;letter-spacing:2px;font-weight:500;">Affiliated to JNTUK &nbsp;|&nbsp; AICTE Approved &nbsp;|&nbsp; NAAC A</div>
</div>
""", unsafe_allow_html=True)

    col_a, col_b, col_c = st.columns(3)

    with col_a:
        st.markdown("""
<div style="background:linear-gradient(145deg,#ffffff,#f0f9ff);border:2px solid #22d3ee;border-radius:16px;padding:1.8rem;text-align:center;height:100%;box-shadow:0 4px 20px rgba(34,211,238,0.2);">
    <div style="font-size:2.5rem;margin-bottom:0.8rem;">🎓</div>
    <div style="font-family:'Rajdhani',sans-serif;font-size:1.3rem;font-weight:700;color:#0891b2;letter-spacing:1px;margin-bottom:0.8rem;">ABOUT THE COLLEGE</div>
    <div style="font-family:'Exo 2',sans-serif;font-size:0.9rem;color:#0c4a6e;line-height:1.7;">
        BVC College of Engineering, Palacharla is an <b style="color:#0891b2;">Autonomous</b> premier technical institution dedicated to excellence in engineering education. Established with a vision to nurture innovation.
    </div>
</div>
""", unsafe_allow_html=True)

    with col_b:
        st.markdown("""
<div style="background:linear-gradient(145deg,#ffffff,#f0f9ff);border:2px solid #22d3ee;border-radius:16px;padding:1.8rem;text-align:center;height:100%;box-shadow:0 4px 20px rgba(34,211,238,0.2);">
    <div style="font-size:2.5rem;margin-bottom:0.8rem;">🔬</div>
    <div style="font-family:'Rajdhani',sans-serif;font-size:1.3rem;font-weight:700;color:#0891b2;letter-spacing:1px;margin-bottom:0.8rem;">ABOUT THE PROJECT</div>
    <div style="font-family:'Exo 2',sans-serif;font-size:0.9rem;color:#0c4a6e;line-height:1.7;">
        NeuroScan AI - B.Tech final year project for early detection of Parkinson's Disease using brain MRI scans. Powered by ResNet+ViT deep learning achieving 99.4% validation accuracy.
    </div>
</div>
""", unsafe_allow_html=True)

    with col_c:
        st.markdown("""
<div style="background:linear-gradient(145deg,#ffffff,#f0f9ff);border:2px solid #22d3ee;border-radius:16px;padding:1.8rem;text-align:center;height:100%;box-shadow:0 4px 20px rgba(34,211,238,0.2);">
    <div style="font-size:2.5rem;margin-bottom:0.8rem;">🤖</div>
    <div style="font-family:'Rajdhani',sans-serif;font-size:1.3rem;font-weight:700;color:#0891b2;letter-spacing:1px;margin-bottom:0.8rem;">TECHNOLOGY STACK</div>
    <div style="font-family:'Exo 2',sans-serif;font-size:0.9rem;color:#0c4a6e;line-height:1.7;">
        <b style="color:#0891b2;">AI Framework:</b> PyTorch + timm<br>
        <b style="color:#0891b2;">Model:</b> ResNet+Vit<br>
        <b style="color:#0891b2;">Frontend:</b> Streamlit<br>
        <b style="color:#0891b2;">Reports:</b> ReportLab PDF<br>
        <b style="color:#0891b2;">Dataset:</b> Parkinson's Brain MRI
    </div>
</div>
""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("""
<div style="font-family:'Rajdhani',sans-serif;font-size:1.5rem;font-weight:700;color:#0c4a6e;letter-spacing:2px;border-left:4px solid #06b6d4;padding-left:14px;margin:1.5rem 0 1rem;">
    👨‍💻 PROJECT TEAM
</div>
""", unsafe_allow_html=True)

    team = [
        {"roll": "236M5A0408", "name": "GAMINI SRINIVASU", "icon": "👨‍💻"},
        {"roll": "226M1A0460", "name": "SALADI ANUSHA", "icon": "👩‍💻"},
        {"roll": "226M1A0473", "name": "V V S VARDHAN", "icon": "👨‍💻"},
        {"roll": "236M5A0415", "name": "N L SANDEEP", "icon": "👨‍💻"},
    ]
    cols = st.columns(4)
    for i, member in enumerate(team):
        with cols[i]:
            st.markdown(f"""
<div style="background:linear-gradient(145deg,#ffffff,#f0f9ff);border:2px solid #22d3ee;border-radius:14px;padding:1.3rem;text-align:center;box-shadow:0 4px 15px rgba(34,211,238,0.2);">
    <div style="font-size:2.5rem;margin-bottom:0.5rem;">{member["icon"]}</div>
    <div style="font-family:'Rajdhani',sans-serif;font-size:1.05rem;font-weight:700;color:#0891b2;margin-bottom:0.3rem;">{member["name"]}</div>
    <div style="font-family:'Exo 2',sans-serif;font-size:0.85rem;color:#0c4a6e;font-weight:600;">{member["roll"]}</div>
</div>
""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── PROJECT GUIDANCE ─────────────────────────────────────────────────────
    st.markdown("""
<div style="font-family:'Rajdhani',sans-serif;font-size:1.5rem;font-weight:700;color:#0c4a6e;letter-spacing:2px;border-left:4px solid #06b6d4;padding-left:14px;margin:1.5rem 0 1rem;">
    👨‍🏫 PROJECT GUIDANCE
</div>
""", unsafe_allow_html=True)

    # ── ROW 1: Project Guide — full-width centred panel ───────────────────────
    st.markdown("""
<div style="background:linear-gradient(135deg,#e0f7ff 0%,#b3ecff 40%,#e8f9ff 100%);border:2.5px solid #06b6d4;border-radius:20px;padding:2.4rem 3rem;text-align:center;box-shadow:0 6px 30px rgba(6,182,212,0.28),inset 0 1px 0 rgba(255,255,255,0.8);position:relative;overflow:hidden;margin-bottom:1.2rem;">
    <div style="position:absolute;top:0;left:10%;right:10%;height:3px;background:linear-gradient(90deg,transparent,#06b6d4,#0891b2,#06b6d4,transparent);border-radius:0 0 4px 4px;"></div>
    <div style="display:inline-block;background:linear-gradient(135deg,#0891b2,#06b6d4);color:#fff;font-family:'Rajdhani',sans-serif;font-size:0.75rem;font-weight:700;letter-spacing:3px;padding:0.28rem 1.1rem;border-radius:50px;margin-bottom:1.2rem;text-transform:uppercase;box-shadow:0 2px 10px rgba(6,182,212,0.4);">⭐ PROJECT GUIDE</div>
    <div style="font-size:3rem;margin-bottom:0.6rem;filter:drop-shadow(0 2px 6px rgba(6,182,212,0.4));">👨‍🏫</div>
    <div style="font-family:'Rajdhani',sans-serif;font-size:1.6rem;font-weight:700;color:#0c4a6e;letter-spacing:2px;margin-bottom:0.4rem;">Ms. N P U V S N PAVAN KUMAR, M.Tech</div>
    <div style="width:60px;height:2px;background:linear-gradient(90deg,#06b6d4,#0891b2);margin:0.6rem auto 0.9rem;border-radius:2px;"></div>
    <div style="display:inline-flex;flex-wrap:wrap;justify-content:center;gap:0.6rem;">
        <span style="background:rgba(6,182,212,0.12);border:1px solid rgba(6,182,212,0.35);border-radius:50px;padding:0.25rem 0.9rem;font-family:'Exo 2',sans-serif;font-size:0.83rem;color:#0369a1;font-weight:600;">Assistant Professor</span>
        <span style="background:rgba(6,182,212,0.12);border:1px solid rgba(6,182,212,0.35);border-radius:50px;padding:0.25rem 0.9rem;font-family:'Exo 2',sans-serif;font-size:0.83rem;color:#0369a1;font-weight:600;">Department of ECE</span>
        <span style="background:rgba(6,182,212,0.12);border:1px solid rgba(6,182,212,0.35);border-radius:50px;padding:0.25rem 0.9rem;font-family:'Exo 2',sans-serif;font-size:0.83rem;color:#0369a1;font-weight:600;">Deputy Controller of Examinations –III</span>
    </div>
    <div style="position:absolute;bottom:0;left:10%;right:10%;height:3px;background:linear-gradient(90deg,transparent,#0891b2,#06b6d4,#0891b2,transparent);border-radius:4px 4px 0 0;"></div>
</div>
""", unsafe_allow_html=True)

    # ── ROW 2: Coordinator + HOD — side by side ───────────────────────────────
    col_g2, col_g3 = st.columns(2, gap="large")

    with col_g2:
        st.markdown("""
<div style="background:linear-gradient(145deg,#f0fffe,#e0f9f5,#f5fffe);border:2px solid #22d3ee;border-radius:18px;padding:2rem;text-align:center;height:100%;box-shadow:0 4px 22px rgba(34,211,238,0.22);position:relative;overflow:hidden;">
    <div style="position:absolute;top:0;left:0;right:0;height:3px;background:linear-gradient(90deg,transparent,#22d3ee,transparent);"></div>
    <div style="display:inline-block;background:linear-gradient(135deg,#0e7490,#22d3ee);color:#fff;font-family:'Rajdhani',sans-serif;font-size:0.72rem;font-weight:700;letter-spacing:2.5px;padding:0.22rem 0.9rem;border-radius:50px;margin-bottom:1rem;text-transform:uppercase;">📋 PROJECT COORDINATOR</div>
    <div style="font-size:2.6rem;margin-bottom:0.5rem;">📋</div>
    <div style="font-family:'Rajdhani',sans-serif;font-size:1.25rem;font-weight:700;color:#0c4a6e;letter-spacing:1.5px;margin-bottom:0.4rem;">Mr. K ANJI BABU, M.Tech</div>
    <div style="width:50px;height:2px;background:linear-gradient(90deg,#22d3ee,#0891b2);margin:0.5rem auto 0.8rem;border-radius:2px;"></div>
    <div style="display:flex;flex-direction:column;gap:0.4rem;align-items:center;">
        <span style="background:rgba(34,211,238,0.1);border:1px solid rgba(34,211,238,0.3);border-radius:50px;padding:0.22rem 0.85rem;font-family:'Exo 2',sans-serif;font-size:0.82rem;color:#0369a1;font-weight:600;">Assistant Professor</span>
        <span style="background:rgba(34,211,238,0.1);border:1px solid rgba(34,211,238,0.3);border-radius:50px;padding:0.22rem 0.85rem;font-family:'Exo 2',sans-serif;font-size:0.82rem;color:#0369a1;font-weight:600;">Department of ECE</span>
    </div>
    <div style="position:absolute;bottom:0;left:0;right:0;height:3px;background:linear-gradient(90deg,transparent,#22d3ee,transparent);"></div>
</div>
""", unsafe_allow_html=True)

    with col_g3:
        st.markdown("""
<div style="background:linear-gradient(145deg,#fff8f0,#fff1e0,#fffaf5);border:2px solid #f59e0b;border-radius:18px;padding:2rem;text-align:center;height:100%;box-shadow:0 4px 22px rgba(245,158,11,0.18);position:relative;overflow:hidden;">
    <div style="position:absolute;top:0;left:0;right:0;height:3px;background:linear-gradient(90deg,transparent,#f59e0b,transparent);"></div>
    <div style="display:inline-block;background:linear-gradient(135deg,#b45309,#f59e0b);color:#fff;font-family:'Rajdhani',sans-serif;font-size:0.72rem;font-weight:700;letter-spacing:2.5px;padding:0.22rem 0.9rem;border-radius:50px;margin-bottom:1rem;text-transform:uppercase;">👨‍💼 HEAD OF DEPARTMENT</div>
    <div style="font-size:2.6rem;margin-bottom:0.5rem;">👨‍💼</div>
    <div style="font-family:'Rajdhani',sans-serif;font-size:1.25rem;font-weight:700;color:#78350f;letter-spacing:1.5px;margin-bottom:0.4rem;">Dr. S A VARA PRASAD</div>
    <div style="width:50px;height:2px;background:linear-gradient(90deg,#f59e0b,#b45309);margin:0.5rem auto 0.8rem;border-radius:2px;"></div>
    <div style="display:flex;flex-direction:column;gap:0.4rem;align-items:center;">
        <span style="background:rgba(245,158,11,0.1);border:1px solid rgba(245,158,11,0.35);border-radius:50px;padding:0.22rem 0.85rem;font-family:'Exo 2',sans-serif;font-size:0.82rem;color:#92400e;font-weight:600;">Head of Department</span>
        <span style="background:rgba(245,158,11,0.1);border:1px solid rgba(245,158,11,0.35);border-radius:50px;padding:0.22rem 0.85rem;font-family:'Exo 2',sans-serif;font-size:0.82rem;color:#92400e;font-weight:600;">Electronics &amp; Communication Engineering</span>
    </div>
    <div style="position:absolute;bottom:0;left:0;right:0;height:3px;background:linear-gradient(90deg,transparent,#f59e0b,transparent);"></div>
</div>
""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col_d1, col_d2 = st.columns(2)

    with col_d1:
        st.markdown("""
<div style="background:linear-gradient(145deg,#ffffff,#f0f9ff);border:2px solid #22d3ee;border-radius:16px;padding:1.8rem;box-shadow:0 4px 20px rgba(34,211,238,0.2);">
    <div style="font-family:'Rajdhani',sans-serif;font-size:1.3rem;font-weight:700;color:#0891b2;letter-spacing:1px;margin-bottom:1rem;">📍 COLLEGE DETAILS</div>
    <table style="width:100%;font-family:'Exo 2',sans-serif;font-size:0.9rem;border-collapse:collapse;">
        <tr><td style="color:#0369a1;padding:0.4rem 0;width:40%;font-weight:600;">Institution</td><td style="color:#0c4a6e;">BVC College of Engineering</td></tr>
        <tr><td style="color:#0369a1;padding:0.4rem 0;font-weight:600;">Status</td><td style="color:#0c4a6e;"><b style="color:#0891b2;">Autonomous</b></td></tr>
        <tr><td style="color:#0369a1;padding:0.4rem 0;font-weight:600;">Location</td><td style="color:#0c4a6e;">Palacharla, Andhra Pradesh</td></tr>
        <tr><td style="color:#0369a1;padding:0.4rem 0;font-weight:600;">University</td><td style="color:#0c4a6e;">JNTU Kakinada (JNTUK)</td></tr>
        <tr><td style="color:#0369a1;padding:0.4rem 0;font-weight:600;">Approval</td><td style="color:#0c4a6e;">AICTE Approved</td></tr>
        <tr><td style="color:#0369a1;padding:0.4rem 0;font-weight:600;">Department</td><td style="color:#0c4a6e;">Electronics & Communication Engineering (ECE)</td></tr>
        <tr><td style="color:#0369a1;padding:0.4rem 0;font-weight:600;">Project Type</td><td style="color:#0c4a6e;">B.Tech Final Year Project</td></tr>
        <tr><td style="color:#0369a1;padding:0.4rem 0;font-weight:600;">Academic Year</td><td style="color:#0c4a6e;">2025 – 2026</td></tr>
    </table>
</div>
""", unsafe_allow_html=True)

    with col_d2:
        st.markdown("""
<div style="background:linear-gradient(145deg,#ffffff,#f0f9ff);border:2px solid #22d3ee;border-radius:16px;padding:1.8rem;box-shadow:0 4px 20px rgba(34,211,238,0.2);">
    <div style="font-family:'Rajdhani',sans-serif;font-size:1.3rem;font-weight:700;color:#0891b2;letter-spacing:1px;margin-bottom:1rem;">📊 PROJECT HIGHLIGHTS</div>
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:1rem;">
        <div style="background:linear-gradient(135deg,#ecfeff,#cffafe);border:2px solid #06b6d4;border-radius:10px;padding:1rem;text-align:center;box-shadow:0 0 10px rgba(6,182,212,0.2);">
            <div style="font-family:'Rajdhani',sans-serif;font-size:1.8rem;font-weight:700;color:#059669;">99.4%</div>
            <div style="font-family:'Exo 2',sans-serif;font-size:0.75rem;color:#0c4a6e;font-weight:600;">Validation Accuracy</div>
        </div>
        <div style="background:linear-gradient(135deg,#ecfeff,#cffafe);border:2px solid #06b6d4;border-radius:10px;padding:1rem;text-align:center;box-shadow:0 0 10px rgba(6,182,212,0.2);">
            <div style="font-family:'Rajdhani',sans-serif;font-size:1.8rem;font-weight:700;color:#0891b2;">831</div>
            <div style="font-family:'Exo 2',sans-serif;font-size:0.75rem;color:#0c4a6e;font-weight:600;">MRI Images</div>
        </div>
        <div style="background:linear-gradient(135deg,#ecfeff,#cffafe);border:2px solid #06b6d4;border-radius:10px;padding:1rem;text-align:center;box-shadow:0 0 10px rgba(6,182,212,0.2);">
            <div style="font-family:'Rajdhani',sans-serif;font-size:1.8rem;font-weight:700;color:#ea580c;">1</div>
            <div style="font-family:'Exo 2',sans-serif;font-size:0.75rem;color:#0c4a6e;font-weight:600;">AI Architecture</div>
        </div>
        <div style="background:linear-gradient(135deg,#ecfeff,#cffafe);border:2px solid #06b6d4;border-radius:10px;padding:1rem;text-align:center;box-shadow:0 0 10px rgba(6,182,212,0.2);">
            <div style="font-family:'Rajdhani',sans-serif;font-size:1.8rem;font-weight:700;color:#dc2626;">2</div>
            <div style="font-family:'Exo 2',sans-serif;font-size:0.75rem;color:#0c4a6e;font-weight:600;">Classes (Normal / PD)</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
<div style="background:linear-gradient(135deg,#ecfeff,#cffafe);border:2px solid #06b6d4;border-radius:14px;padding:1.5rem 2rem;text-align:center;box-shadow:0 0 15px rgba(6,182,212,0.3);">
    <div style="font-family:'Exo 2',sans-serif;font-size:0.9rem;color:#0c4a6e;letter-spacing:1px;font-weight:500;">
        ⚕️ This project is for <b style="color:#0891b2;">academic and research purposes only</b>. 
        It is not intended to replace clinical medical diagnosis. 
        Always consult a qualified neurologist for medical advice.
    </div>
</div>
""", unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown(
    '<div style="text-align:center;padding:2rem 1rem 1rem;">'
    '<div style="font-size:0.75rem;letter-spacing:3px;color:#3a6a8a;text-transform:uppercase;margin-bottom:0.5rem;">Research &amp; Educational Use Only &nbsp;&middot;&nbsp; Not for Clinical Diagnosis</div>'
    '<div style="font-size:0.85rem;color:#4a90b8;letter-spacing:1px;">NeuroScan AI &nbsp;&middot;&nbsp; ResNet-ViT &nbsp;&middot;&nbsp; Built with PyTorch &amp; Streamlit</div>'
    '</div>',
    unsafe_allow_html=True
)