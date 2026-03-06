import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
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
import torchvision.models as models
import base64
import os
import gdown
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd

# ── GOOGLE DRIVE CONFIG ───────────────────────────────────────────────────────
GDRIVE_FILE_ID = "10aU1zRciaAzlYyhv-dkrXZmzJLJnom65"
MODEL_PATH = "new_vit.pth"

def download_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("⬇️ Downloading AI model — first time only, please wait…"):
            try:
                url = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}"
                gdown.download(url, MODEL_PATH, quiet=False)
                st.success("✅ Model downloaded!")
            except Exception as e:
                st.error(f"❌ Download failed: {e}")
                st.info("💡 Make sure the Google Drive file is shared as 'Anyone with the link'")
                st.stop()

# ── MODEL ARCHITECTURE (matches Colab exactly) ────────────────────────────────
class ResNetBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet50(weights="IMAGENET1K_V1")
        for param in resnet.parameters():
            param.requires_grad = False
        for param in resnet.layer4.parameters():
            param.requires_grad = True
        self.features = nn.Sequential(*list(resnet.children())[:-2])

    def forward(self, x):
        return self.features(x)


class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, 1)

    def forward(self, x):
        x = self.proj(x)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim=768, heads=8, depth=6):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=heads,
            dim_feedforward=2048, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, depth)

    def forward(self, x):
        return self.encoder(x)


class ResNetViT(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone      = ResNetBackbone()
        self.patch_embed   = PatchEmbedding()
        self.transformer   = TransformerEncoder()
        self.norm          = nn.LayerNorm(768)
        self.dropout       = nn.Dropout(0.3)
        self.fc            = nn.Linear(768, num_classes)

    def forward(self, x):
        x = self.backbone(x)
        x = self.patch_embed(x)
        x = self.transformer(x)
        x = x.mean(dim=1)
        x = self.norm(x)
        x = self.dropout(x)
        x = self.fc(x)
        return x


# ── GRAD-CAM ──────────────────────────────────────────────────────────────────
class GradCAM:
    def __init__(self, model):
        self.model       = model
        self.gradients   = None
        self.activations = None
        self._hooks      = []

        # Target layer4 inside the ResNetBackbone Sequential
        # features[-1] is layer4 (the last block before avgpool/fc)
        # We hook the last Bottleneck block inside layer4
        target_layer = None
        for name, module in model.backbone.features.named_children():
            target_layer = module   # keeps updating → ends at layer4
        # target_layer is now layer4; hook its last Bottleneck
        last_bottleneck = list(target_layer.children())[-1]

        self._hooks.append(
            last_bottleneck.register_forward_hook(self._save_activation)
        )
        self._hooks.append(
            last_bottleneck.register_full_backward_hook(self._save_gradient)
        )

    def _save_activation(self, module, inp, out):
        self.activations = out.detach()

    def _save_gradient(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def generate(self, img_tensor, class_idx):
        self.model.eval()
        img_tensor = img_tensor.clone().requires_grad_(True)
        output = self.model(img_tensor)
        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0][class_idx] = 1
        output.backward(gradient=one_hot, retain_graph=True)

        # Global average pool gradients → weights
        weights = self.gradients.mean(dim=[2, 3], keepdim=True)   # (1, C, 1, 1)
        cam     = (weights * self.activations).sum(dim=1, keepdim=True)  # (1,1,H,W)
        cam     = F.relu(cam)

        # Normalize per-sample so small activations don't vanish
        cam_min = cam.min()
        cam_max = cam.max()
        cam     = (cam - cam_min) / (cam_max - cam_min + 1e-8)

        cam = F.interpolate(cam, size=(224, 224), mode='bilinear', align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        return cam

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()


def apply_colormap_on_image(original_img, activation_map):
    """Overlay Grad-CAM heatmap on original image."""
    heatmap = cm.jet(activation_map)[:, :, :3]
    heatmap = (heatmap * 255).astype(np.uint8)
    orig    = np.array(original_img.resize((224, 224))).astype(np.float32)
    overlay = (0.55 * orig + 0.45 * heatmap.astype(np.float32))
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)
    return Image.fromarray(overlay), Image.fromarray(heatmap)


# ── HELPERS ───────────────────────────────────────────────────────────────────
def get_logo_base64(path):
    try:
        if os.path.exists(path):
            with open(path, "rb") as f:
                data = base64.b64encode(f.read()).decode()
            ext  = path.split(".")[-1].lower()
            mime = "image/jpeg" if ext in ["jpg", "jpeg"] else f"image/{ext}"
            return f"data:{mime};base64,{data}"
    except:
        pass
    return None


TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


@st.cache_resource
def load_model():
    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    model      = ResNetViT(num_classes=2)
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    return model, device


def predict_single(model, device, image: Image.Image):
    img_rgb    = image.convert("RGB")
    img_tensor = TRANSFORM(img_rgb).unsqueeze(0).to(device)

    model.eval()
    gcam = GradCAM(model)

    # Run forward + backward with gradients enabled
    with torch.enable_grad():
        img_tensor = img_tensor.requires_grad_(True)
        outputs    = model(img_tensor)
        probs      = F.softmax(outputs, dim=1)
        conf, predicted = torch.max(probs, 1)
        class_idx  = predicted.item()

    cam_map = gcam.generate(img_tensor, class_idx)
    gcam.remove_hooks()

    overlay, heatmap = apply_colormap_on_image(img_rgb, cam_map)

    return {
        "prediction":     ["Normal", "Parkinson's Disease"][class_idx],
        "confidence":     float(conf.item() * 100),
        "normal_prob":    float(probs[0][0].item() * 100),
        "parkinson_prob": float(probs[0][1].item() * 100),
        "cam_overlay":    overlay,
        "cam_heatmap":    heatmap,
        "timestamp":      datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "image":          img_rgb,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="NeuroScan AI — Parkinson's MRI Analysis",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@500;600;700&family=Exo+2:wght@300;400;600;700;800&family=Share+Tech+Mono:wght@400&display=swap');

:root {
  --cyan:       #00d4ff;
  --cyan-dim:   rgba(0,212,255,0.18);
  --cyan-glow:  rgba(0,212,255,0.45);
  --blue:       #0052cc;
  --blue-mid:   #0099e6;
  --green:      #00e676;
  --red:        #ff4444;
  --amber:      #ffb300;
  --bg-deep:    #020817;
  --bg-card:    #0a1628;
  --text-main:  #e0f4ff;
  --text-mute:  #7ecfee;
}

@keyframes borderRun {
  0%   { background-position: 0% 50%; }
  100% { background-position: 200% 50%; }
}
@keyframes fadeUp {
  from { opacity:0; transform:translateY(22px); }
  to   { opacity:1; transform:translateY(0); }
}
@keyframes neonFlicker {
  0%,19%,21%,23%,25%,54%,56%,100% {
    text-shadow: 0 0 10px var(--cyan), 0 0 30px var(--cyan), 0 0 60px var(--cyan-glow);
  }
  20%,24%,55% { text-shadow: none; }
}
@keyframes bgPan {
  0%   { background-position: 0% 0%; }
  100% { background-position: 100% 100%; }
}
@keyframes scanPulse {
  0%,100% { opacity:.03; }
  50%      { opacity:.07; }
}
@keyframes shimmerBar {
  0%   { left:-100%; }
  100% { left:200%;  }
}

html, body, .stApp {
  background: var(--bg-deep) !important;
  font-family: 'Exo 2', sans-serif !important;
  color: var(--text-main) !important;
}

/* Animated grid overlay */
.stApp::before {
  content:'';
  position:fixed; inset:0;
  background-image:
    linear-gradient(rgba(0,212,255,0.025) 1px, transparent 1px),
    linear-gradient(90deg, rgba(0,212,255,0.025) 1px, transparent 1px);
  background-size:55px 55px;
  pointer-events:none; z-index:0;
  animation: bgPan 80s linear infinite;
}
/* Ambient glow */
.stApp::after {
  content:'';
  position:fixed; top:-15%; left:50%; transform:translateX(-50%);
  width:1000px; height:700px;
  background:radial-gradient(ellipse at center,
    rgba(0,82,204,0.10) 0%,
    rgba(0,212,255,0.05) 40%,
    transparent 70%);
  pointer-events:none; z-index:0;
}

.block-container {
  padding-top:1rem !important;
  max-width:1340px !important;
  position:relative; z-index:1;
}
#MainMenu, footer, header { visibility:hidden; }

/* ─── Neo Card ─────────────────────────────────── */
.neo-card {
  background: linear-gradient(145deg,#0d1b2e 0%,#0a1628 60%,#071220 100%);
  border: 1px solid rgba(0,212,255,0.22);
  border-radius: 20px;
  padding: 1.8rem 2rem;
  margin: 0.8rem 0;
  box-shadow:
    0 4px 40px rgba(0,0,0,.7),
    0 0 80px rgba(0,82,204,.07),
    inset 0 1px 0 rgba(255,255,255,.06);
  position:relative; overflow:hidden;
  animation: fadeUp .5s cubic-bezier(.22,1,.36,1);
  transition: border-color .3s, box-shadow .3s;
}
.neo-card:hover {
  border-color: rgba(0,212,255,.4);
  box-shadow:
    0 8px 60px rgba(0,0,0,.7),
    0 0 100px rgba(0,82,204,.13),
    inset 0 1px 0 rgba(255,255,255,.08);
}
.neo-card::before {
  content:'';
  position:absolute; top:0; left:0; right:0; height:1px;
  background:linear-gradient(90deg,
    transparent 0%,
    rgba(0,212,255,.8) 40%,
    rgba(0,212,255,1)  50%,
    rgba(0,212,255,.8) 60%,
    transparent 100%);
}
.neo-card::after {
  content:'';
  position:absolute; top:0; left:0;
  width:36px; height:36px;
  border-top:2px solid rgba(0,212,255,.5);
  border-left:2px solid rgba(0,212,255,.5);
  border-radius:20px 0 0 0;
}
/* Bottom-right corner accent */
.neo-card .corner-br {
  position:absolute; bottom:0; right:0;
  width:36px; height:36px;
  border-bottom:2px solid rgba(0,212,255,.35);
  border-right:2px solid rgba(0,212,255,.35);
  border-radius:0 0 20px 0;
}

/* ─── Inputs ───────────────────────────────────── */
.stTextInput input, .stNumberInput input,
.stTextArea textarea, .stDateInput input {
  background:linear-gradient(145deg,#04101e,#061020) !important;
  color:var(--text-main) !important;
  border:1px solid rgba(0,180,220,.3) !important;
  border-radius:10px !important;
  font-family:'Exo 2',sans-serif !important;
  transition:border-color .25s, box-shadow .25s !important;
}
.stTextInput input:focus, .stNumberInput input:focus,
.stTextArea textarea:focus {
  border-color:var(--cyan) !important;
  box-shadow:0 0 0 3px rgba(0,212,255,.13) !important;
}
div[data-baseweb="select"] > div {
  background:linear-gradient(145deg,#04101e,#061020) !important;
  border:1px solid rgba(0,180,220,.3) !important;
  border-radius:10px !important;
  color:var(--text-main) !important;
}
label, .stSelectbox label, .stTextInput label,
.stNumberInput label, .stDateInput label, .stTextArea label {
  color:var(--text-mute) !important;
  font-family:'Exo 2',sans-serif !important;
  font-size:.8rem !important;
  font-weight:700 !important;
  letter-spacing:1.2px !important;
  text-transform:uppercase !important;
}

/* ─── Buttons ──────────────────────────────────── */
.stButton > button {
  background:linear-gradient(135deg,#003fa3 0%,#0066cc 50%,#0099e6 100%) !important;
  color:#fff !important;
  border:1px solid rgba(0,180,255,.4) !important;
  border-radius:12px !important;
  padding:.8rem 1.4rem !important;
  font-family:'Rajdhani',sans-serif !important;
  font-size:1.1rem !important; font-weight:700 !important;
  letter-spacing:3px !important; text-transform:uppercase !important;
  width:100% !important;
  transition:all .3s cubic-bezier(.34,1.56,.64,1) !important;
  box-shadow:0 4px 20px rgba(0,100,200,.5),inset 0 1px 0 rgba(255,255,255,.15) !important;
  position:relative !important; overflow:hidden !important;
}
.stButton > button::before {
  content:'' !important; position:absolute !important;
  top:0 !important; left:-100% !important;
  width:100% !important; height:100% !important;
  background:linear-gradient(90deg,transparent,rgba(255,255,255,.12),transparent) !important;
  transition:left .5s ease !important;
}
.stButton > button:hover::before { left:100% !important; }
.stButton > button:hover {
  background:linear-gradient(135deg,#0052cc 0%,#0088dd 50%,#00c0ff 100%) !important;
  transform:translateY(-3px) scale(1.01) !important;
  box-shadow:0 10px 32px rgba(0,140,255,.6),0 0 40px rgba(0,212,255,.2) !important;
}
.stDownloadButton > button {
  background:linear-gradient(135deg,#004d25 0%,#007a40 50%,#00aa55 100%) !important;
  color:#fff !important;
  border:1px solid rgba(0,200,100,.35) !important;
  border-radius:12px !important;
  padding:.8rem 1.4rem !important;
  font-family:'Rajdhani',sans-serif !important;
  font-size:1.1rem !important; font-weight:700 !important;
  letter-spacing:3px !important; text-transform:uppercase !important;
  width:100% !important;
  transition:all .3s cubic-bezier(.34,1.56,.64,1) !important;
  box-shadow:0 4px 20px rgba(0,160,80,.45) !important;
}
.stDownloadButton > button:hover {
  transform:translateY(-3px) scale(1.01) !important;
  box-shadow:0 10px 30px rgba(0,200,100,.55) !important;
}

/* ─── File Uploader ────────────────────────────── */
[data-testid="stFileUploader"] {
  background:linear-gradient(145deg,#060f1c,#091522) !important;
  border:2px dashed rgba(0,212,255,.35) !important;
  border-radius:18px !important; padding:1.6rem !important;
  transition:all .35s ease !important;
}
[data-testid="stFileUploader"]:hover {
  border-color:var(--cyan) !important;
  box-shadow:0 0 40px rgba(0,212,255,.12) !important;
}

/* ─── Result Cards ─────────────────────────────── */
.result-card {
  background:linear-gradient(145deg,#071424,#0c1f38,#081830);
  border:1px solid rgba(0,212,255,.28);
  border-radius:18px; padding:1.4rem 1.8rem;
  margin:.4rem 0; text-align:center;
  position:relative; overflow:hidden;
  transition:transform .3s ease, box-shadow .3s ease;
  animation:fadeUp .55s cubic-bezier(.22,1,.36,1);
}
.result-card:hover { transform:translateY(-3px); }
.result-card::before {
  content:''; position:absolute;
  top:0; left:20%; right:20%; height:1px;
  background:linear-gradient(90deg,transparent,rgba(0,212,255,.6),transparent);
}
.result-card::after {
  content:''; position:absolute;
  bottom:0; left:0; right:0; height:3px;
  background:linear-gradient(90deg,#0052cc,#00d4ff,#00ff88,#00d4ff,#0052cc);
  background-size:300% 100%;
  animation:borderRun 2.5s linear infinite;
}
.result-label {
  font-family:'Share Tech Mono',monospace;
  font-size:.75rem; color:var(--text-mute);
  letter-spacing:3px; text-transform:uppercase; margin-bottom:.5rem;
}
.result-value-normal {
  font-family:'Rajdhani',sans-serif; font-size:2.4rem; font-weight:700;
  color:var(--green);
  text-shadow:0 0 20px rgba(0,230,118,.7),0 0 60px rgba(0,230,118,.25);
}
.result-value-parkinson {
  font-family:'Rajdhani',sans-serif; font-size:2.4rem; font-weight:700;
  color:var(--red);
  text-shadow:0 0 20px rgba(255,68,68,.7),0 0 60px rgba(255,68,68,.25);
}
.result-value-confidence {
  font-family:'Rajdhani',sans-serif; font-size:2.4rem; font-weight:700;
  color:var(--cyan);
  text-shadow:0 0 20px var(--cyan-glow),0 0 60px rgba(0,212,255,.2);
  animation:neonFlicker 8s infinite;
}

/* ─── Image Frame ──────────────────────────────── */
.img-frame {
  border:1px solid rgba(0,212,255,.3);
  border-radius:16px; padding:.7rem;
  background:#040e1c;
  box-shadow:0 0 40px rgba(0,212,255,.1),inset 0 0 20px rgba(0,0,0,.6);
  position:relative;
}
.img-frame::before {
  content:''; position:absolute;
  top:5px; left:5px; width:20px; height:20px;
  border-top:2px solid var(--cyan); border-left:2px solid var(--cyan);
  border-radius:3px 0 0 0;
}
.img-frame::after {
  content:''; position:absolute;
  bottom:5px; right:5px; width:20px; height:20px;
  border-bottom:2px solid var(--cyan); border-right:2px solid var(--cyan);
  border-radius:0 0 3px 0;
}

/* ─── Batch Row ────────────────────────────────── */
.batch-row {
  background:linear-gradient(135deg,#071424,#0c1f38);
  border:1px solid rgba(0,212,255,.2);
  border-radius:14px; padding:1rem 1.4rem; margin:.5rem 0;
  display:flex; align-items:center; gap:1.2rem;
  animation:fadeUp .4s ease;
}
.batch-normal  { border-left:4px solid var(--green); }
.batch-parkinson { border-left:4px solid var(--red); }

/* ─── Metrics ──────────────────────────────────── */
[data-testid="stMetric"] {
  background:linear-gradient(145deg,#071424,#0c1f38) !important;
  border:1px solid rgba(0,212,255,.18) !important;
  border-radius:14px !important; padding:1rem 1.4rem !important;
  transition:transform .25s !important;
}
[data-testid="stMetric"]:hover { transform:translateY(-2px) !important; }
[data-testid="stMetricValue"] {
  font-family:'Rajdhani',sans-serif !important;
  font-size:1.8rem !important; font-weight:700 !important;
  color:var(--cyan) !important;
  text-shadow:0 0 12px rgba(0,212,255,.4) !important;
}
[data-testid="stMetricLabel"] {
  color:var(--text-mute) !important;
  font-family:'Share Tech Mono',monospace !important;
  font-size:.72rem !important; letter-spacing:1.5px !important;
  text-transform:uppercase !important;
}

/* ─── Tabs ─────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
  background:linear-gradient(135deg,#040e1c,#061424) !important;
  border:1px solid rgba(0,212,255,.15) !important;
  border-radius:14px !important; padding:.4rem !important; gap:.3rem !important;
}
.stTabs [data-baseweb="tab"] {
  font-family:'Rajdhani',sans-serif !important;
  font-size:1rem !important; font-weight:700 !important;
  letter-spacing:2px !important; text-transform:uppercase !important;
  color:var(--text-mute) !important; border-radius:10px !important;
  transition:all .25s ease !important;
}
.stTabs [data-baseweb="tab"]:hover { color:var(--cyan) !important; }
.stTabs [aria-selected="true"] {
  background:linear-gradient(135deg,rgba(0,82,204,.4),rgba(0,212,255,.18)) !important;
  color:#fff !important;
  border:1px solid rgba(0,212,255,.3) !important;
}
.stTabs [data-baseweb="tab-highlight"],
.stTabs [data-baseweb="tab-border"] { display:none !important; }

/* ─── Sidebar ──────────────────────────────────── */
[data-testid="stSidebar"] {
  background:linear-gradient(180deg,#020c18,#030e1e) !important;
  border-right:1px solid rgba(0,212,255,.12) !important;
}
[data-testid="stSidebar"] * { font-family:'Exo 2',sans-serif !important; }

/* ─── Progress ─────────────────────────────────── */
.stProgress > div > div > div {
  background:linear-gradient(90deg,var(--blue),var(--cyan)) !important;
  border-radius:10px !important;
  box-shadow:0 0 10px rgba(0,212,255,.4) !important;
}

/* ─── Scrollbar ────────────────────────────────── */
::-webkit-scrollbar { width:5px; }
::-webkit-scrollbar-track { background:#020817; }
::-webkit-scrollbar-thumb {
  background:linear-gradient(180deg,#0052cc,#00d4ff);
  border-radius:10px;
}
hr {
  border:none !important; height:1px !important;
  background:linear-gradient(90deg,
    transparent,rgba(0,82,204,.4),rgba(0,212,255,.7),rgba(0,82,204,.4),transparent) !important;
  margin:2rem 0 !important;
}
/* Section heading */
.section-title {
  font-family:'Rajdhani',sans-serif;
  font-size:1.45rem; font-weight:700; color:#fff;
  letter-spacing:3px; text-transform:uppercase;
  border-left:4px solid var(--cyan); padding-left:14px;
  margin:1.6rem 0 1rem; position:relative;
}
.section-title::after {
  content:''; position:absolute;
  bottom:-6px; left:18px;
  width:36px; height:2px;
  background:var(--cyan); border-radius:2px;
}
/* Heatmap legend */
.heatmap-legend {
  display:flex; gap:.5rem; align-items:center;
  font-family:'Share Tech Mono',monospace;
  font-size:.72rem; color:var(--text-mute); margin-top:.5rem;
}
.legend-bar {
  flex:1; height:10px; border-radius:5px;
  background:linear-gradient(90deg,#00008b,#0000ff,#00ff00,#ffff00,#ff0000);
}
</style>
""", unsafe_allow_html=True)

# ── HERO HEADER ───────────────────────────────────────────────────────────────
logo_src = get_logo_base64("logo.png")
logo_html = (
    f'<img src="{logo_src}" style="width:88px;height:88px;object-fit:contain;border-radius:50%;'
    f'border:2px solid rgba(0,212,255,.5);box-shadow:0 0 28px rgba(0,212,255,.3);margin:0 auto 1.1rem;display:block;"/>'
    if logo_src else
    '<div style="width:88px;height:88px;background:radial-gradient(circle,#0a2040,#020c18);'
    'border:2px solid rgba(0,212,255,.5);border-radius:50%;display:flex;align-items:center;'
    'justify-content:center;margin:0 auto 1.1rem;font-size:2.8rem;">&#x1F9E0;</div>'
)

st.markdown(
    '<div style="background:linear-gradient(180deg,#020c18,#041428);border-bottom:1px solid rgba(0,212,255,.22);'
    'padding:2.5rem 2rem 1.8rem;text-align:center;position:relative;overflow:hidden;">'
    '<div style="width:70px;height:3px;margin:0 auto 1.2rem;background:linear-gradient(90deg,#0052cc,#00d4ff);border-radius:2px;"></div>'
    + logo_html +
    '<div style="font-family:Rajdhani,sans-serif;font-size:3.6rem;font-weight:700;color:#fff;letter-spacing:8px;'
    'text-transform:uppercase;line-height:1;margin-bottom:.5rem;text-shadow:0 0 40px rgba(0,180,255,.4);">'
    'NEUROSCAN <span style="color:#00d4ff;">AI</span></div>'
    '<div style="font-family:\'Exo 2\',sans-serif;font-size:.95rem;color:#7ecfee;letter-spacing:4px;'
    'text-transform:uppercase;margin-bottom:1.4rem;">Parkinson\'s Disease &nbsp;·&nbsp; MRI Analysis &nbsp;·&nbsp; Deep Learning</div>'
    '<div style="display:inline-flex;gap:2rem;background:rgba(0,212,255,.06);border:1px solid rgba(0,212,255,.18);'
    'border-radius:50px;padding:.55rem 1.8rem;font-size:.8rem;letter-spacing:1px;color:#94d8ff;">'
    '<span>🤖 ResNet-ViT</span>'
    '<span style="color:rgba(0,212,255,.3);">|</span>'
    '<span>🎯 99.4% Accuracy</span>'
    '<span style="color:rgba(0,212,255,.3);">|</span>'
    '<span>🔥 Grad-CAM</span>'
    '<span style="color:rgba(0,212,255,.3);">|</span>'
    '<span>⚡ Batch Analysis</span>'
    '</div>'
    '<div style="width:70px;height:3px;margin:1.2rem auto 0;background:linear-gradient(90deg,#00d4ff,#0052cc);border-radius:2px;"></div>'
    '</div>',
    unsafe_allow_html=True,
)

# ── MODEL DOWNLOAD ────────────────────────────────────────────────────────────
download_model()

# ── SESSION STATE ─────────────────────────────────────────────────────────────
for key, default in [
    ("prediction_made", False),
    ("patient_data", {}),
    ("prediction_result", {}),
    ("batch_results", []),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🔬 NeuroScan AI")
    st.info("Advanced AI system for brain MRI analysis. Upload a scan to detect Parkinson's Disease in seconds.")
    st.markdown("---")
    st.markdown("### 🤖 Model Info")
    st.success(
        "**Architecture:** ResNet50 + ViT\n\n"
        "**Classes:** Normal / Parkinson's\n\n"
        "**Accuracy:** ~99.4%\n\n"
        "**Status:** 🟢 Online"
    )
    st.markdown("---")
    st.markdown("### 🔥 What's New")
    st.markdown(
        "- ✅ Grad-CAM heatmap\n"
        "- ✅ Batch analysis\n"
        "- ✅ Fixed architecture\n"
        "- ✅ PDF reports"
    )
    st.markdown("---")
    st.markdown("### ⚠️ Disclaimer")
    st.warning("For research/academic purposes only. Not a substitute for clinical diagnosis.")
    st.markdown("---")
    col_s1, col_s2 = st.columns(2)
    with col_s1: st.metric("Precision", "100%")
    with col_s2: st.metric("Recall",    "98%")

# ── MAIN TABS ─────────────────────────────────────────────────────────────────
tab_scan, tab_batch, tab_about = st.tabs([
    "🧠  Single MRI Analysis",
    "📦  Batch Analysis",
    "🏫  About Us",
])

# ══════════════════════════════════════════════════════════════════════════════
#  TAB 1 — SINGLE MRI ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
with tab_scan:
    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.markdown('<div class="section-title">👤 Patient Information</div>', unsafe_allow_html=True)
        st.markdown("<div class='neo-card'>", unsafe_allow_html=True)
        patient_name   = st.text_input("📝 Full Name",   placeholder="Patient's full name")
        c_age, c_gen   = st.columns(2)
        with c_age: patient_age    = st.number_input("🎂 Age", 0, 120, 30)
        with c_gen: patient_gender = st.selectbox("⚧ Gender", ["Male", "Female", "Other"])
        patient_id     = st.text_input("🆔 Patient ID",  placeholder="e.g. P-2024-0001")
        scan_date      = st.date_input("📅 Scan Date",   value=datetime.now())
        medical_history= st.text_area("📋 Medical History (Optional)", height=90)
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="section-title">🖼️ MRI Image Upload</div>', unsafe_allow_html=True)
        st.markdown("<div class='neo-card'>", unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "🔬 Upload Brain MRI Scan",
            type=["png", "jpg", "jpeg"],
            help="PNG / JPG / JPEG",
        )
        if uploaded_file:
            image = Image.open(uploaded_file)
            st.markdown("<div class='img-frame'>", unsafe_allow_html=True)
            st.image(image, caption="📸 Uploaded MRI", use_column_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.info("⬆️ Upload a brain MRI scan to begin")
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div class="section-title">🔍 AI Analysis & Results</div>', unsafe_allow_html=True)

    col_btn, col_report = st.columns([1, 1], gap="large")

    with col_btn:
        if st.button("🚀 ANALYZE MRI SCAN"):
            if not patient_name or not patient_id or not uploaded_file:
                st.error("⚠️ Fill in all required fields and upload an MRI scan.")
            else:
                with st.spinner("🧠 Analyzing with AI…"):
                    st.session_state.patient_data = {
                        "name": patient_name, "age": patient_age,
                        "gender": patient_gender, "patient_id": patient_id,
                        "scan_date": scan_date.strftime("%Y-%m-%d"),
                        "medical_history": medical_history,
                    }
                    try:
                        model, device = load_model()
                        result = predict_single(model, device, image)
                        st.session_state.prediction_result = result
                        st.session_state.prediction_made   = True
                        st.success("✅ Analysis complete!")
                        st.balloons()
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Error: {e}")

    # ── RESULTS ───────────────────────────────────────────────────────────────
    if st.session_state.prediction_made:
        r        = st.session_state.prediction_result
        is_normal = r["prediction"] == "Normal"
        val_class = "result-value-normal" if is_normal else "result-value-parkinson"
        risk_icon = "✅" if is_normal else "⚠️"

        # Diagnosis + Confidence
        st.markdown("<div class='neo-card'>", unsafe_allow_html=True)
        st.markdown("### 📊 Diagnostic Results")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(
                f"<div class='result-card'><div class='result-label'>🎯 Diagnosis</div>"
                f"<div class='{val_class}'>{r['prediction']}</div></div>",
                unsafe_allow_html=True,
            )
        with c2:
            st.markdown(
                f"<div class='result-card'><div class='result-label'>💯 Confidence</div>"
                f"<div class='result-value-confidence'>{r['confidence']:.1f}%</div></div>",
                unsafe_allow_html=True,
            )
        with c3:
            st.metric("✅ Normal",      f"{r['normal_prob']:.1f}%")
        with c4:
            st.metric("⚠️ Parkinson's", f"{r['parkinson_prob']:.1f}%")

        # Probability bars
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("**Normal probability**")
        st.progress(r["normal_prob"] / 100)
        st.markdown("**Parkinson's probability**")
        st.progress(r["parkinson_prob"] / 100)
        st.markdown(
            f"<p style='color:#7ecfee;font-size:.82rem;margin-top:.8rem;'>⏰ {r['timestamp']}</p>",
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

        # ── GRAD-CAM ─────────────────────────────────────────────────────────
        st.markdown('<div class="section-title">🔥 Grad-CAM Heatmap</div>', unsafe_allow_html=True)
        st.markdown("<div class='neo-card'>", unsafe_allow_html=True)
        st.markdown(
            "<p style='color:#7ecfee;font-size:.88rem;'>Grad-CAM highlights the brain regions "
            "that most influenced the model's prediction. <strong style='color:#00d4ff;'>Warmer "
            "colours (red/yellow)</strong> = higher attention.</p>",
            unsafe_allow_html=True,
        )
        gc1, gc2, gc3 = st.columns(3)
        with gc1:
            st.markdown("<div class='img-frame'>", unsafe_allow_html=True)
            st.image(r["image"], caption="Original MRI", use_column_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
        with gc2:
            st.markdown("<div class='img-frame'>", unsafe_allow_html=True)
            st.image(r["cam_heatmap"], caption="🌡️ Heatmap", use_column_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
        with gc3:
            st.markdown("<div class='img-frame'>", unsafe_allow_html=True)
            st.image(r["cam_overlay"], caption="🔥 Overlay", use_column_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
        st.markdown(
            "<div class='heatmap-legend'>"
            "<span>Low</span><div class='legend-bar'></div><span>High</span>"
            "</div>",
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with col_report:
        if st.session_state.prediction_made:
            if st.button("📄 GENERATE PDF REPORT"):
                with st.spinner("📝 Building report…"):
                    try:
                        buffer = io.BytesIO()
                        doc    = SimpleDocTemplate(buffer, pagesize=letter)
                        story  = []
                        styles = getSampleStyleSheet()

                        title_s   = ParagraphStyle("T", parent=styles["Heading1"], fontSize=22,
                            textColor=colors.HexColor("#1565c0"), spaceAfter=24,
                            alignment=TA_CENTER, fontName="Helvetica-Bold")
                        heading_s = ParagraphStyle("H", parent=styles["Heading2"], fontSize=15,
                            textColor=colors.HexColor("#1976d2"), spaceAfter=10,
                            spaceBefore=10, fontName="Helvetica-Bold")

                        story.append(Paragraph("NEUROSCAN AI — PARKINSON'S MRI REPORT", title_s))
                        story.append(Spacer(1, 14))

                        story.append(Paragraph("Patient Information", heading_s))
                        p = st.session_state.patient_data
                        pt = Table([
                            ["Field","Details"],
                            ["Patient Name", p["name"]],
                            ["Patient ID",   p["patient_id"]],
                            ["Age",          str(p["age"])],
                            ["Gender",       p["gender"]],
                            ["Scan Date",    p["scan_date"]],
                        ], colWidths=[2*inch, 4*inch])
                        pt.setStyle(TableStyle([
                            ("BACKGROUND",   (0,0),(-1,0), colors.HexColor("#1565c0")),
                            ("TEXTCOLOR",    (0,0),(-1,0), colors.whitesmoke),
                            ("FONTNAME",     (0,0),(-1,0), "Helvetica-Bold"),
                            ("FONTSIZE",     (0,0),(-1,0), 12),
                            ("BOTTOMPADDING",(0,0),(-1,0), 10),
                            ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white, colors.lightgrey]),
                            ("GRID",         (0,0),(-1,-1), 1, colors.black),
                            ("FONTNAME",     (0,1),(-1,-1), "Helvetica"),
                            ("FONTSIZE",     (0,1),(-1,-1), 10),
                        ]))
                        story.append(pt)
                        story.append(Spacer(1, 14))

                        if p["medical_history"]:
                            story.append(Paragraph("Medical History", heading_s))
                            story.append(Paragraph(p["medical_history"], styles["Normal"]))
                            story.append(Spacer(1, 14))

                        story.append(Paragraph("AI Analysis Results", heading_s))
                        r = st.session_state.prediction_result
                        rt = Table([
                            ["Metric","Value"],
                            ["Diagnosis",             r["prediction"]],
                            ["Confidence",            f"{r['confidence']:.2f}%"],
                            ["Normal Probability",    f"{r['normal_prob']:.2f}%"],
                            ["Parkinson's Probability",f"{r['parkinson_prob']:.2f}%"],
                            ["Analysis Time",         r["timestamp"]],
                            ["AI Model",              "ResNet-ViT (Custom)"],
                        ], colWidths=[2.5*inch, 3.5*inch])
                        rt.setStyle(TableStyle([
                            ("BACKGROUND",   (0,0),(-1,0), colors.HexColor("#1565c0")),
                            ("TEXTCOLOR",    (0,0),(-1,0), colors.whitesmoke),
                            ("FONTNAME",     (0,0),(-1,0), "Helvetica-Bold"),
                            ("FONTSIZE",     (0,0),(-1,0), 12),
                            ("BOTTOMPADDING",(0,0),(-1,0), 10),
                            ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white, colors.lightgrey]),
                            ("GRID",         (0,0),(-1,-1), 1, colors.black),
                            ("FONTNAME",     (0,1),(-1,-1), "Helvetica"),
                            ("FONTSIZE",     (0,1),(-1,-1), 10),
                        ]))
                        story.append(rt)
                        story.append(Spacer(1, 14))

                        story.append(Paragraph("Brain MRI Scan", heading_s))
                        img_buf = io.BytesIO()
                        r["image"].save(img_buf, format="PNG")
                        img_buf.seek(0)
                        story.append(RLImage(img_buf, width=3.5*inch, height=3.5*inch))
                        story.append(Spacer(1, 8))

                        story.append(Paragraph("Grad-CAM Heatmap Overlay", heading_s))
                        cam_buf = io.BytesIO()
                        r["cam_overlay"].save(cam_buf, format="PNG")
                        cam_buf.seek(0)
                        story.append(RLImage(cam_buf, width=3.5*inch, height=3.5*inch))
                        story.append(Spacer(1, 14))

                        story.append(Paragraph("Disclaimer", heading_s))
                        story.append(Paragraph(
                            "This report is generated by an AI system for research/educational purposes only. "
                            "It must NOT replace clinical diagnosis by qualified medical professionals.",
                            styles["Normal"],
                        ))
                        story.append(Spacer(1, 20))
                        footer_s = ParagraphStyle("F", parent=styles["Normal"], fontSize=8,
                            textColor=colors.grey, alignment=TA_CENTER)
                        story.append(Paragraph(
                            f"NeuroScan AI — BVC College of Engineering, Palacharla — "
                            f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                            footer_s,
                        ))
                        doc.build(story)
                        buffer.seek(0)
                        st.download_button(
                            "⬇️ DOWNLOAD PDF REPORT",
                            data=buffer,
                            file_name=f"NeuroScan_{p['patient_id']}_{datetime.now().strftime('%Y%m%d')}.pdf",
                            mime="application/pdf",
                        )
                        st.success("✅ PDF ready!")
                    except Exception as e:
                        st.error(f"❌ PDF error: {e}")
        else:
            st.info("📋 Run analysis first to generate a report.")


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 2 — BATCH ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
with tab_batch:
    st.markdown('<div class="section-title">📦 Batch MRI Analysis</div>', unsafe_allow_html=True)
    st.markdown(
        "<p style='color:#7ecfee;'>Upload multiple MRI scans at once. "
        "Each image is analysed independently and results are summarised below.</p>",
        unsafe_allow_html=True,
    )

    batch_files = st.file_uploader(
        "🔬 Upload Multiple Brain MRI Scans",
        type=["png", "jpg", "jpeg"],
        accept_multiple_files=True,
        help="Select multiple files at once",
        key="batch_uploader",
    )

    if batch_files:
        st.info(f"📁 {len(batch_files)} file(s) selected")

        if st.button("🚀 RUN BATCH ANALYSIS"):
            try:
                model, device = load_model()
            except Exception as e:
                st.error(f"❌ Model load error: {e}")
                st.stop()

            batch_results = []
            progress_bar  = st.progress(0)
            status_text   = st.empty()

            for i, f in enumerate(batch_files):
                status_text.markdown(
                    f"<p style='color:#00d4ff;font-family:Share Tech Mono,monospace;'>"
                    f"Analysing {f.name} ({i+1}/{len(batch_files)})…</p>",
                    unsafe_allow_html=True,
                )
                img = Image.open(f)
                res = predict_single(model, device, img)
                res["filename"] = f.name
                batch_results.append(res)
                progress_bar.progress((i + 1) / len(batch_files))

            st.session_state.batch_results = batch_results
            status_text.empty()
            st.success(f"✅ Batch analysis complete — {len(batch_results)} scans processed!")
            st.rerun()

    # ── BATCH RESULTS ─────────────────────────────────────────────────────────
    if st.session_state.batch_results:
        results = st.session_state.batch_results
        n_total    = len(results)
        n_normal   = sum(1 for r in results if r["prediction"] == "Normal")
        n_parkinson = n_total - n_normal
        avg_conf   = np.mean([r["confidence"] for r in results])

        # Summary metrics
        st.markdown('<div class="section-title">📊 Batch Summary</div>', unsafe_allow_html=True)
        m1, m2, m3, m4 = st.columns(4)
        with m1: st.metric("🔢 Total Scans",   n_total)
        with m2: st.metric("✅ Normal",         n_normal)
        with m3: st.metric("⚠️ Parkinson's",   n_parkinson)
        with m4: st.metric("💯 Avg Confidence", f"{avg_conf:.1f}%")

        # Pie chart
        st.markdown('<div class="section-title">📈 Distribution</div>', unsafe_allow_html=True)
        chart_col, table_col = st.columns([1, 1], gap="large")

        with chart_col:
            fig, ax = plt.subplots(figsize=(5, 4), facecolor="#0a1628")
            wedge_colors = ["#00e676", "#ff4444"]
            wedges, texts, autotexts = ax.pie(
                [n_normal, n_parkinson],
                labels=["Normal", "Parkinson's"],
                autopct="%1.0f%%",
                colors=wedge_colors,
                startangle=90,
                wedgeprops=dict(edgecolor="#0a1628", linewidth=2),
                textprops=dict(color="#e0f4ff", fontsize=13),
            )
            for at in autotexts:
                at.set_color("#0a1628")
                at.set_fontweight("bold")
            ax.set_facecolor("#0a1628")
            ax.set_title("Batch Distribution", color="#00d4ff", fontsize=14, pad=12)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        with table_col:
            df = pd.DataFrame([{
                "File":        r["filename"],
                "Prediction":  r["prediction"],
                "Confidence":  f"{r['confidence']:.1f}%",
                "Normal %":    f"{r['normal_prob']:.1f}%",
                "Parkinson's %": f"{r['parkinson_prob']:.1f}%",
            } for r in results])
            st.dataframe(df, use_container_width=True, height=300)

            # CSV download
            csv = df.to_csv(index=False)
            st.download_button(
                "⬇️ DOWNLOAD CSV REPORT",
                data=csv,
                file_name=f"batch_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
            )

        # Per-image cards
        st.markdown('<div class="section-title">🖼️ Per-Image Results</div>', unsafe_allow_html=True)
        for r in results:
            is_n    = r["prediction"] == "Normal"
            border  = "batch-normal" if is_n else "batch-parkinson"
            icon    = "✅" if is_n else "⚠️"
            colour  = "#00e676" if is_n else "#ff4444"
            cols    = st.columns([1, 2, 2, 2])
            with cols[0]:
                st.markdown("<div class='img-frame'>", unsafe_allow_html=True)
                st.image(r["image"], use_column_width=True)
                st.markdown("</div>", unsafe_allow_html=True)
            with cols[1]:
                st.markdown(
                    f"<p style='font-family:Rajdhani,sans-serif;font-size:1rem;"
                    f"color:#94d8ff;margin-bottom:.3rem;'>📄 {r['filename']}</p>"
                    f"<p style='font-family:Rajdhani,sans-serif;font-size:1.6rem;"
                    f"font-weight:700;color:{colour};margin:0;'>{icon} {r['prediction']}</p>"
                    f"<p style='font-family:Share Tech Mono,monospace;font-size:.8rem;"
                    f"color:#7ecfee;margin-top:.3rem;'>{r['timestamp']}</p>",
                    unsafe_allow_html=True,
                )
            with cols[2]:
                st.metric("Confidence",   f"{r['confidence']:.1f}%")
                st.metric("Normal %",     f"{r['normal_prob']:.1f}%")
            with cols[3]:
                st.image(r["cam_overlay"], caption="Grad-CAM", use_column_width=True)
            st.markdown("<hr style='margin:.8rem 0;'>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 3 — ABOUT US
# ══════════════════════════════════════════════════════════════════════════════
with tab_about:
    college_logo_src = get_logo_base64("bvcr.jpg")
    about_logo_html  = (
        f'<img src="{college_logo_src}" style="width:110px;height:110px;object-fit:contain;'
        f'margin-bottom:1rem;filter:drop-shadow(0 0 12px rgba(6,182,212,0.5));"/>'
        if college_logo_src
        else '<div style="font-size:3.5rem;margin-bottom:1rem;">🏛️</div>'
    )

    st.markdown(f"""
<div style="background:linear-gradient(135deg,#f0f9ff,#e0f2fe);border:2px solid #06b6d4;
border-radius:24px;padding:3rem;margin-bottom:2rem;text-align:center;
box-shadow:0 0 30px rgba(6,182,212,0.3);">
  {about_logo_html}
  <div style="font-family:Rajdhani,sans-serif;font-size:2.4rem;font-weight:700;
  color:#0c4a6e;letter-spacing:3px;margin-bottom:.4rem;">BVC COLLEGE OF ENGINEERING</div>
  <div style="font-family:'Exo 2',sans-serif;font-size:1.1rem;color:#0891b2;
  letter-spacing:4px;text-transform:uppercase;font-weight:600;">AUTONOMOUS</div>
  <div style="width:80px;height:3px;background:linear-gradient(90deg,#06b6d4,#0891b2,#06b6d4);
  margin:1rem auto;border-radius:2px;"></div>
  <div style="font-family:'Exo 2',sans-serif;font-size:.9rem;color:#0c4a6e;letter-spacing:2px;">
    Affiliated to JNTUK &nbsp;|&nbsp; AICTE Approved &nbsp;|&nbsp; NAAC A
  </div>
</div>
""", unsafe_allow_html=True)

    ca, cb, cc = st.columns(3)
    for col, icon, title, body in [
        (ca, "🎓", "ABOUT THE COLLEGE",
         "BVC College of Engineering, Palacharla is an <b style='color:#0891b2;'>Autonomous</b> "
         "premier technical institution dedicated to excellence in engineering education."),
        (cb, "🔬", "ABOUT THE PROJECT",
         "NeuroScan AI — B.Tech final year project for early Parkinson's detection from brain MRI. "
         "Powered by ResNet+ViT, achieving 99.4% validation accuracy."),
        (cc, "🤖", "TECHNOLOGY STACK",
         "<b style='color:#0891b2;'>AI:</b> PyTorch<br>"
         "<b style='color:#0891b2;'>Model:</b> ResNet-ViT<br>"
         "<b style='color:#0891b2;'>XAI:</b> Grad-CAM<br>"
         "<b style='color:#0891b2;'>Frontend:</b> Streamlit<br>"
         "<b style='color:#0891b2;'>Reports:</b> ReportLab"),
    ]:
        with col:
            st.markdown(f"""
<div style="background:linear-gradient(145deg,#fff,#f0f9ff);border:2px solid #22d3ee;
border-radius:16px;padding:1.8rem;text-align:center;height:100%;
box-shadow:0 4px 20px rgba(34,211,238,0.2);">
  <div style="font-size:2.4rem;margin-bottom:.7rem;">{icon}</div>
  <div style="font-family:Rajdhani,sans-serif;font-size:1.2rem;font-weight:700;
  color:#0891b2;letter-spacing:1px;margin-bottom:.7rem;">{title}</div>
  <div style="font-family:'Exo 2',sans-serif;font-size:.88rem;color:#0c4a6e;line-height:1.7;">{body}</div>
</div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div style="font-family:Rajdhani,sans-serif;font-size:1.4rem;font-weight:700;'
                'color:#0c4a6e;letter-spacing:2px;border-left:4px solid #06b6d4;padding-left:14px;'
                'margin:1.5rem 0 1rem;">👨‍💻 PROJECT TEAM</div>', unsafe_allow_html=True)

    team = [
        {"roll": "236M5A0408", "name": "G SRINIVASU",        "icon": "👨‍💻"},
        {"roll": "226M1A0460", "name": "S ANUSHA DEVI",      "icon": "👩‍💻"},
        {"roll": "226M1A0473", "name": "V V SIVA VARDHAN",   "icon": "👨‍💻"},
        {"roll": "236M5A0415", "name": "N L SANDEEP",        "icon": "👨‍💻"},
    ]
    t_cols = st.columns(4)
    for i, m in enumerate(team):
        with t_cols[i]:
            st.markdown(f"""
<div style="background:linear-gradient(145deg,#fff,#f0f9ff);border:2px solid #22d3ee;
border-radius:14px;padding:1.3rem;text-align:center;
box-shadow:0 4px 15px rgba(34,211,238,0.2);">
  <div style="font-size:2.4rem;margin-bottom:.4rem;">{m['icon']}</div>
  <div style="font-family:Rajdhani,sans-serif;font-size:1rem;font-weight:700;color:#0891b2;margin-bottom:.2rem;">{m['name']}</div>
  <div style="font-family:'Exo 2',sans-serif;font-size:.82rem;color:#0c4a6e;font-weight:600;">{m['roll']}</div>
</div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div style="font-family:Rajdhani,sans-serif;font-size:1.4rem;font-weight:700;'
                'color:#0c4a6e;letter-spacing:2px;border-left:4px solid #06b6d4;padding-left:14px;'
                'margin:1.5rem 0 1rem;">👨‍🏫 PROJECT GUIDANCE</div>', unsafe_allow_html=True)

    # Guide
    st.markdown("""
<div style="background:linear-gradient(135deg,#e0f7ff,#b3ecff,#e8f9ff);border:2.5px solid #06b6d4;
border-radius:20px;padding:2.2rem 3rem;text-align:center;
box-shadow:0 6px 30px rgba(6,182,212,0.28);margin-bottom:1.2rem;">
  <div style="display:inline-block;background:linear-gradient(135deg,#0891b2,#06b6d4);color:#fff;
  font-family:Rajdhani,sans-serif;font-size:.75rem;font-weight:700;letter-spacing:3px;
  padding:.28rem 1.1rem;border-radius:50px;margin-bottom:1rem;">⭐ PROJECT GUIDE</div>
  <div style="font-size:3rem;margin-bottom:.5rem;">👨‍🏫</div>
  <div style="font-family:Rajdhani,sans-serif;font-size:1.5rem;font-weight:700;
  color:#0c4a6e;letter-spacing:2px;margin-bottom:.6rem;">Ms. N P U V S N PAVAN KUMAR, M.Tech</div>
  <div style="display:inline-flex;flex-wrap:wrap;justify-content:center;gap:.6rem;">
    <span style="background:rgba(6,182,212,0.12);border:1px solid rgba(6,182,212,0.35);
    border-radius:50px;padding:.22rem .9rem;font-family:'Exo 2',sans-serif;
    font-size:.82rem;color:#0369a1;font-weight:600;">Assistant Professor</span>
    <span style="background:rgba(6,182,212,0.12);border:1px solid rgba(6,182,212,0.35);
    border-radius:50px;padding:.22rem .9rem;font-family:'Exo 2',sans-serif;
    font-size:.82rem;color:#0369a1;font-weight:600;">Department of ECE</span>
    <span style="background:rgba(6,182,212,0.12);border:1px solid rgba(6,182,212,0.35);
    border-radius:50px;padding:.22rem .9rem;font-family:'Exo 2',sans-serif;
    font-size:.82rem;color:#0369a1;font-weight:600;">Deputy Controller of Examinations – III</span>
  </div>
</div>
""", unsafe_allow_html=True)

    g2, g3 = st.columns(2, gap="large")
    with g2:
        st.markdown("""
<div style="background:linear-gradient(145deg,#f0fffe,#e0f9f5);border:2px solid #22d3ee;
border-radius:18px;padding:2rem;text-align:center;
box-shadow:0 4px 22px rgba(34,211,238,0.22);">
  <div style="display:inline-block;background:linear-gradient(135deg,#0e7490,#22d3ee);color:#fff;
  font-family:Rajdhani,sans-serif;font-size:.72rem;font-weight:700;letter-spacing:2.5px;
  padding:.22rem .9rem;border-radius:50px;margin-bottom:1rem;">📋 PROJECT COORDINATOR</div>
  <div style="font-size:2.5rem;margin-bottom:.4rem;">📋</div>
  <div style="font-family:Rajdhani,sans-serif;font-size:1.2rem;font-weight:700;color:#0c4a6e;margin-bottom:.5rem;">Mr. K ANJI BABU, M.Tech</div>
  <span style="background:rgba(34,211,238,0.1);border:1px solid rgba(34,211,238,0.3);
  border-radius:50px;padding:.2rem .8rem;font-family:'Exo 2',sans-serif;
  font-size:.8rem;color:#0369a1;font-weight:600;">Assistant Professor · ECE</span>
</div>""", unsafe_allow_html=True)
    with g3:
        st.markdown("""
<div style="background:linear-gradient(145deg,#fff8f0,#fff1e0);border:2px solid #f59e0b;
border-radius:18px;padding:2rem;text-align:center;
box-shadow:0 4px 22px rgba(245,158,11,0.18);">
  <div style="display:inline-block;background:linear-gradient(135deg,#b45309,#f59e0b);color:#fff;
  font-family:Rajdhani,sans-serif;font-size:.72rem;font-weight:700;letter-spacing:2.5px;
  padding:.22rem .9rem;border-radius:50px;margin-bottom:1rem;">👨‍💼 HEAD OF DEPARTMENT</div>
  <div style="font-size:2.5rem;margin-bottom:.4rem;">👨‍💼</div>
  <div style="font-family:Rajdhani,sans-serif;font-size:1.2rem;font-weight:700;color:#78350f;margin-bottom:.5rem;">Dr. S A VARA PRASAD, Ph.D, M.Tech</div>
  <div style="display:flex;flex-direction:column;gap:.4rem;align-items:center;">
    <span style="background:rgba(245,158,11,0.1);border:1px solid rgba(245,158,11,0.35);
    border-radius:50px;padding:.2rem .8rem;font-family:'Exo 2',sans-serif;
    font-size:.8rem;color:#92400e;font-weight:600;">Professor &amp; HOD</span>
    <span style="background:rgba(245,158,11,0.1);border:1px solid rgba(245,158,11,0.35);
    border-radius:50px;padding:.2rem .8rem;font-family:'Exo 2',sans-serif;
    font-size:.8rem;color:#92400e;font-weight:600;">Chairman of BoS &amp; Anti Ragging Committee</span>
    <span style="background:rgba(245,158,11,0.1);border:1px solid rgba(245,158,11,0.35);
    border-radius:50px;padding:.2rem .8rem;font-family:'Exo 2',sans-serif;
    font-size:.8rem;color:#92400e;font-weight:600;">Electronics &amp; Communication Engineering</span>
  </div>
</div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
<div style="background:linear-gradient(135deg,#ecfeff,#cffafe);border:2px solid #06b6d4;
border-radius:14px;padding:1.4rem 2rem;text-align:center;
box-shadow:0 0 15px rgba(6,182,212,0.3);">
  <div style="font-family:'Exo 2',sans-serif;font-size:.9rem;color:#0c4a6e;letter-spacing:1px;">
    ⚕️ This project is for <b style="color:#0891b2;">academic and research purposes only</b>.
    Always consult a qualified neurologist for medical advice.
  </div>
</div>""", unsafe_allow_html=True)

# ── FOOTER ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    '<div style="text-align:center;padding:1.5rem 1rem .8rem;">'
    '<div style="font-size:.72rem;letter-spacing:3px;color:#3a6a8a;text-transform:uppercase;margin-bottom:.4rem;">'
    'Research &amp; Educational Use Only · Not for Clinical Diagnosis</div>'
    '<div style="font-size:.82rem;color:#4a90b8;letter-spacing:1px;">'
    'NeuroScan AI · ResNet-ViT · Grad-CAM · Built with PyTorch &amp; Streamlit</div>'
    '</div>',
    unsafe_allow_html=True,
)
