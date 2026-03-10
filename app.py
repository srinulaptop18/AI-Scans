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
from reportlab.platypus import (
    SimpleDocTemplate, Table, TableStyle,
    Paragraph, Spacer, Image as RLImage,
)
from reportlab.lib.enums import TA_CENTER
import torchvision.transforms as transforms
import torchvision.models as models
import base64
import os
import gdown
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd

# ══════════════════════════════════════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════════════════════════════════════
GDRIVE_FILE_ID = "1p2uIwGMGI06iPyuHYeqUAw2EtBN53vvq"
MODEL_PATH     = "new_ntau.pth"


# ══════════════════════════════════════════════════════════════════════════════
#  MODEL DOWNLOAD
# ══════════════════════════════════════════════════════════════════════════════
def download_model() -> None:
    if not os.path.exists(MODEL_PATH):
        with st.spinner("⬇️ Downloading AI model — first time only, please wait…"):
            try:
                gdown.download(
                    f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}",
                    MODEL_PATH, quiet=False,
                )
                st.success("✅ Model downloaded!")
            except Exception as exc:
                st.error(f"❌ Download failed: {exc}")
                st.info("Make sure the Google Drive file is shared as 'Anyone with the link'.")
                st.stop()


# ══════════════════════════════════════════════════════════════════════════════
#  MODEL ARCHITECTURE
# ══════════════════════════════════════════════════════════════════════════════
class ResNetBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet50(weights="IMAGENET1K_V1")
        for p in resnet.parameters():
            p.requires_grad = False
        for p in resnet.layer4.parameters():
            p.requires_grad = True
        self.features = nn.Sequential(*list(resnet.children())[:-2])

    def forward(self, x):
        return self.features(x)


class PatchEmbedding(nn.Module):
    def __init__(self, in_channels: int = 2048, embed_dim: int = 768):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, 1)

    def forward(self, x):
        x = self.proj(x)
        B, C, H, W = x.shape
        return x.flatten(2).transpose(1, 2)


class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim: int = 768, heads: int = 8, depth: int = 6):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=heads,
            dim_feedforward=2048, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, depth)

    def forward(self, x):
        return self.encoder(x)


class ResNetViT(nn.Module):
    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.backbone    = ResNetBackbone()
        self.patch_embed = PatchEmbedding()
        self.transformer = TransformerEncoder()
        self.norm        = nn.LayerNorm(768)
        self.dropout     = nn.Dropout(0.3)
        self.fc          = nn.Linear(768, num_classes)

    def forward(self, x):
        x = self.backbone(x)
        x = self.patch_embed(x)
        x = self.transformer(x)
        x = x.mean(dim=1)
        x = self.norm(x)
        x = self.dropout(x)
        return self.fc(x)


# ══════════════════════════════════════════════════════════════════════════════
#  GRAD-CAM
# ══════════════════════════════════════════════════════════════════════════════
class GradCAM:
    def __init__(self, model: nn.Module):
        self.model       = model
        self.gradients   = None
        self.activations = None
        self._hooks: list = []

        layer4     = model.backbone.features[-1]
        last_block = list(layer4.children())[-1]
        target     = last_block.conv3

        self._hooks.append(target.register_forward_hook(self._save_activation))
        self._hooks.append(target.register_full_backward_hook(self._save_gradient))

    def _save_activation(self, _m, _i, out):
        self.activations = out.detach()

    def _save_gradient(self, _m, _gi, grad_out):
        self.gradients = grad_out[0].detach()

    def generate(self, img_tensor: torch.Tensor, class_idx: int) -> np.ndarray:
        img_tensor = img_tensor.clone().requires_grad_(True)
        with torch.enable_grad():
            output = self.model(img_tensor)
            self.model.zero_grad()
            output[0, class_idx].backward(retain_graph=True)

        if self.gradients is None or self.activations is None:
            return np.zeros((224, 224))

        weights = F.relu(self.gradients).mean(dim=[2, 3], keepdim=True)
        cam     = (weights * self.activations).sum(dim=1, keepdim=True)
        cam     = F.relu(cam)
        cam     = F.interpolate(cam, size=(224, 224), mode="bicubic", align_corners=False)
        cam     = cam.squeeze().cpu().numpy()

        lo, hi = cam.min(), cam.max()
        if hi - lo > 1e-8:
            cam = (cam - lo) / (hi - lo)
        else:
            return np.zeros((224, 224))

        return np.power(cam, 1.8)

    def remove_hooks(self) -> None:
        for h in self._hooks:
            h.remove()


def apply_colormap(orig: Image.Image, cam: np.ndarray):
    heatmap = (cm.jet(cam)[:, :, :3] * 255).astype(np.uint8)
    orig_arr = np.array(orig.resize((224, 224))).astype(np.float32)
    overlay  = np.clip(0.55 * orig_arr + 0.45 * heatmap.astype(np.float32), 0, 255).astype(np.uint8)
    return Image.fromarray(overlay), Image.fromarray(heatmap)


# ══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════════════════
TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def get_logo_b64(path: str) -> str | None:
    try:
        if os.path.exists(path):
            with open(path, "rb") as f:
                data = base64.b64encode(f.read()).decode()
            ext  = path.rsplit(".", 1)[-1].lower()
            mime = "image/jpeg" if ext in ("jpg", "jpeg") else f"image/{ext}"
            return f"data:{mime};base64,{data}"
    except Exception:
        pass
    return None


@st.cache_resource(show_spinner=False)
def load_model():
    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=True)
    model      = ResNetViT(num_classes=2)
    model.load_state_dict(checkpoint)
    model.to(device).eval()
    return model, device


def predict_single(model: nn.Module, device: torch.device, image: Image.Image) -> dict:
    img_rgb    = image.convert("RGB")
    img_tensor = TRANSFORM(img_rgb).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        outputs        = model(img_tensor)
        probs          = F.softmax(outputs, dim=1)
        conf, predicted = torch.max(probs, 1)
        class_idx      = predicted.item()

    gcam    = GradCAM(model)
    cam_map = gcam.generate(img_tensor.clone(), class_idx)
    gcam.remove_hooks()

    overlay, heatmap = apply_colormap(img_rgb, cam_map)

    # Risk level helper
    confidence_pct = float(conf.item() * 100)
    if class_idx == 0:
        risk = "Low"
    else:
        risk = "High" if confidence_pct >= 85 else "Moderate"

    return {
        "prediction":     ["Normal", "Parkinson's Disease"][class_idx],
        "class_idx":      class_idx,
        "confidence":     confidence_pct,
        "normal_prob":    float(probs[0][0].item() * 100),
        "parkinson_prob": float(probs[0][1].item() * 100),
        "risk_level":     risk,
        "cam_overlay":    overlay,
        "cam_heatmap":    heatmap,
        "timestamp":      datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "image":          img_rgb,
    }


def build_pdf(patient: dict, result: dict) -> bytes:
    """Generate a clean, professional PDF report and return raw bytes."""
    buf    = io.BytesIO()
    doc    = SimpleDocTemplate(buf, pagesize=letter,
                               leftMargin=0.75*inch, rightMargin=0.75*inch,
                               topMargin=0.75*inch, bottomMargin=0.75*inch)
    story  = []
    styles = getSampleStyleSheet()

    title_s = ParagraphStyle("T", parent=styles["Heading1"], fontSize=20,
        textColor=colors.HexColor("#0d2b5e"), spaceAfter=6,
        alignment=TA_CENTER, fontName="Helvetica-Bold")
    sub_s   = ParagraphStyle("S", parent=styles["Normal"], fontSize=10,
        textColor=colors.HexColor("#3a7bd5"), spaceAfter=20,
        alignment=TA_CENTER)
    head_s  = ParagraphStyle("H", parent=styles["Heading2"], fontSize=13,
        textColor=colors.HexColor("#0d2b5e"), spaceAfter=8,
        spaceBefore=14, fontName="Helvetica-Bold",
        borderPad=(0, 0, 4, 0))
    body_s  = ParagraphStyle("B", parent=styles["Normal"], fontSize=10,
        textColor=colors.HexColor("#1a1a2e"), leading=14)
    warn_s  = ParagraphStyle("W", parent=styles["Normal"], fontSize=9,
        textColor=colors.HexColor("#7f1d1d"), leading=13,
        backColor=colors.HexColor("#fff1f1"),
        borderPad=6)

    # Title
    story.append(Paragraph("NeuroScan AI", title_s))
    story.append(Paragraph("Parkinson's Disease MRI Analysis Report", sub_s))

    def tbl(rows, col_widths):
        t = Table(rows, colWidths=col_widths)
        t.setStyle(TableStyle([
            ("BACKGROUND",    (0, 0), (-1, 0), colors.HexColor("#0d2b5e")),
            ("TEXTCOLOR",     (0, 0), (-1, 0), colors.white),
            ("FONTNAME",      (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE",      (0, 0), (-1, 0), 11),
            ("BOTTOMPADDING", (0, 0), (-1, 0), 9),
            ("TOPPADDING",    (0, 0), (-1, 0), 9),
            ("ROWBACKGROUNDS",(0, 1), (-1, -1),
             [colors.HexColor("#f0f6ff"), colors.white]),
            ("GRID",          (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
            ("FONTNAME",      (0, 1), (-1, -1), "Helvetica"),
            ("FONTSIZE",      (0, 1), (-1, -1), 10),
            ("TOPPADDING",    (0, 1), (-1, -1), 7),
            ("BOTTOMPADDING", (0, 1), (-1, -1), 7),
            ("LEFTPADDING",   (0, 0), (-1, -1), 10),
        ]))
        return t

    # Patient info
    story.append(Paragraph("Patient Information", head_s))
    story.append(tbl([
        ["Field", "Details"],
        ["Patient Name",    patient["name"]],
        ["Patient ID",      patient["patient_id"]],
        ["Age",             str(patient["age"])],
        ["Gender",          patient["gender"]],
        ["Scan Date",       patient["scan_date"]],
        ["Referring Doctor",patient.get("doctor", "—")],
    ], [2*inch, 4.5*inch]))

    if patient.get("medical_history", "").strip():
        story.append(Spacer(1, 8))
        story.append(Paragraph("Medical History", head_s))
        story.append(Paragraph(patient["medical_history"], body_s))

    # AI results
    story.append(Paragraph("AI Analysis Results", head_s))
    diag_color = colors.HexColor("#14532d") if result["class_idx"] == 0 else colors.HexColor("#7f1d1d")
    results_tbl = tbl([
        ["Metric",                 "Value"],
        ["Diagnosis",              result["prediction"]],
        ["Confidence Score",       f"{result['confidence']:.2f}%"],
        ["Normal Probability",     f"{result['normal_prob']:.2f}%"],
        ["Parkinson's Probability",f"{result['parkinson_prob']:.2f}%"],
        ["Risk Level",             result["risk_level"]],
        ["Analysis Time",          result["timestamp"]],
        ["AI Model",               "ResNet50 + Vision Transformer (ViT)"],
    ], [2.5*inch, 4*inch])
    story.append(results_tbl)

    story.append(Spacer(1, 14))

    # Images side by side
    story.append(Paragraph("Brain MRI Scan & Grad-CAM Heatmap", head_s))
    img_buf = io.BytesIO(); result["image"].save(img_buf, "PNG"); img_buf.seek(0)
    cam_buf = io.BytesIO(); result["cam_overlay"].save(cam_buf, "PNG"); cam_buf.seek(0)
    img_tbl = Table([[
        RLImage(img_buf,  width=2.8*inch, height=2.8*inch),
        RLImage(cam_buf,  width=2.8*inch, height=2.8*inch),
    ]], colWidths=[3.2*inch, 3.2*inch])
    img_tbl.setStyle(TableStyle([
        ("ALIGN",       (0, 0), (-1, -1), "CENTER"),
        ("VALIGN",      (0, 0), (-1, -1), "MIDDLE"),
        ("INNERGRID",   (0, 0), (-1, -1), 0, colors.white),
        ("BOX",         (0, 0), (-1, -1), 0, colors.white),
    ]))
    story.append(img_tbl)
    caption_s = ParagraphStyle("C", parent=styles["Normal"], fontSize=8,
        textColor=colors.grey, alignment=TA_CENTER, spaceBefore=4)
    story.append(Paragraph("Left: Original MRI &nbsp;&nbsp;|&nbsp;&nbsp; Right: Grad-CAM Attention Overlay", caption_s))

    story.append(Spacer(1, 16))
    story.append(Paragraph(
        "⚠️ DISCLAIMER: This report is generated by an AI system for research and educational "
        "purposes only. It must NOT replace clinical diagnosis by a qualified medical professional. "
        "Always consult a licensed neurologist for any medical decisions.",
        warn_s,
    ))

    story.append(Spacer(1, 20))
    footer_s = ParagraphStyle("F", parent=styles["Normal"], fontSize=8,
        textColor=colors.grey, alignment=TA_CENTER)
    story.append(Paragraph(
        f"NeuroScan AI · BVC College of Engineering, Palacharla · "
        f"Generated {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        footer_s,
    ))

    doc.build(story)
    buf.seek(0)
    return buf.read()


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="NeuroScan AI — Parkinson's MRI Analysis",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════════════
#  CSS  — clean dark-clinical theme, no generic purple gradients
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
/* ── Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,wght@0,300;0,400;0,500;0,600;0,700;1,400&family=Space+Grotesk:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');

/* ── Tokens ── */
:root {
  --bg:          #080d14;
  --bg1:         #0d1520;
  --bg2:         #111d2e;
  --bg3:         #162135;
  --border:      rgba(96,165,250,0.14);
  --border-hi:   rgba(96,165,250,0.38);
  --blue:        #3b82f6;
  --blue-dim:    rgba(59,130,246,0.12);
  --blue-glow:   rgba(59,130,246,0.28);
  --teal:        #2dd4bf;
  --teal-dim:    rgba(45,212,191,0.10);
  --green:       #4ade80;
  --red:         #f87171;
  --amber:       #fbbf24;
  --text:        #e2eaf4;
  --text-2:      #8ba8c8;
  --text-3:      #4a6a8a;
  --radius:      14px;
  --radius-sm:   8px;
  --shadow:      0 2px 24px rgba(0,0,0,.55);
}

/* ── Base ── */
html, body, .stApp {
  background: var(--bg) !important;
  font-family: 'DM Sans', sans-serif !important;
  color: var(--text) !important;
}

/* Subtle grid */
.stApp::before {
  content: '';
  position: fixed; inset: 0;
  background-image:
    linear-gradient(rgba(59,130,246,.018) 1px, transparent 1px),
    linear-gradient(90deg, rgba(59,130,246,.018) 1px, transparent 1px);
  background-size: 48px 48px;
  pointer-events: none; z-index: 0;
}

.block-container {
  padding-top: 0 !important;
  max-width: 1360px !important;
  position: relative; z-index: 1;
}
#MainMenu, footer, header { visibility: hidden; }

/* ── Cards ── */
.card {
  background: linear-gradient(160deg, var(--bg2) 0%, var(--bg1) 100%);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 1.6rem 1.8rem;
  margin: .6rem 0;
  box-shadow: var(--shadow);
  position: relative;
  transition: border-color .25s, box-shadow .25s;
}
.card:hover {
  border-color: var(--border-hi);
  box-shadow: 0 4px 40px rgba(59,130,246,.10);
}
.card-inset {
  background: rgba(0,0,0,.25);
  border: 1px solid var(--border);
  border-radius: var(--radius-sm);
  padding: 1.2rem 1.4rem;
  margin: .4rem 0;
}

/* ── Section header ── */
.sec-head {
  display: flex; align-items: center; gap: .7rem;
  font-family: 'Space Grotesk', sans-serif;
  font-size: .72rem; font-weight: 600; color: var(--blue);
  letter-spacing: 2.5px; text-transform: uppercase;
  margin: 1.6rem 0 .8rem;
}
.sec-head::before {
  content: '';
  display: inline-block; width: 3px; height: 16px;
  background: var(--blue); border-radius: 2px;
}
.sec-head::after {
  content: '';
  flex: 1; height: 1px;
  background: linear-gradient(90deg, var(--border-hi), transparent);
}

/* ── Inputs ── */
.stTextInput input, .stNumberInput input,
.stTextArea textarea, .stDateInput input {
  background: var(--bg1) !important;
  color: var(--text) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius-sm) !important;
  font-family: 'DM Sans', sans-serif !important;
  font-size: .93rem !important;
  transition: border-color .2s, box-shadow .2s !important;
}
.stTextInput input:focus, .stNumberInput input:focus,
.stTextArea textarea:focus {
  border-color: var(--blue) !important;
  box-shadow: 0 0 0 3px rgba(59,130,246,.15) !important;
  outline: none !important;
}
div[data-baseweb="select"] > div {
  background: var(--bg1) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius-sm) !important;
  color: var(--text) !important;
}
label {
  color: var(--text-2) !important;
  font-family: 'DM Sans', sans-serif !important;
  font-size: .8rem !important;
  font-weight: 600 !important;
  letter-spacing: .4px !important;
}

/* ── Buttons ── */
.stButton > button {
  background: linear-gradient(135deg, #1d4ed8, #3b82f6) !important;
  color: #fff !important;
  border: 1px solid rgba(96,165,250,.35) !important;
  border-radius: var(--radius-sm) !important;
  padding: .7rem 1.2rem !important;
  font-family: 'Space Grotesk', sans-serif !important;
  font-size: .9rem !important; font-weight: 600 !important;
  letter-spacing: 1.2px !important; text-transform: uppercase !important;
  width: 100% !important;
  transition: all .25s ease !important;
  box-shadow: 0 2px 12px rgba(59,130,246,.35) !important;
}
.stButton > button:hover {
  background: linear-gradient(135deg, #2563eb, #60a5fa) !important;
  transform: translateY(-2px) !important;
  box-shadow: 0 6px 20px rgba(59,130,246,.45) !important;
}
.stButton > button:active { transform: translateY(0) !important; }
.stDownloadButton > button {
  background: linear-gradient(135deg, #065f46, #059669) !important;
  color: #fff !important;
  border: 1px solid rgba(52,211,153,.3) !important;
  border-radius: var(--radius-sm) !important;
  padding: .7rem 1.2rem !important;
  font-family: 'Space Grotesk', sans-serif !important;
  font-size: .9rem !important; font-weight: 600 !important;
  letter-spacing: 1.2px !important; text-transform: uppercase !important;
  width: 100% !important;
  transition: all .25s ease !important;
  box-shadow: 0 2px 12px rgba(5,150,105,.35) !important;
}
.stDownloadButton > button:hover {
  transform: translateY(-2px) !important;
  box-shadow: 0 6px 20px rgba(5,150,105,.45) !important;
}

/* ── File uploader ── */
[data-testid="stFileUploader"] {
  background: var(--bg1) !important;
  border: 2px dashed rgba(59,130,246,.3) !important;
  border-radius: var(--radius) !important;
  padding: 1.4rem !important;
  transition: border-color .25s, box-shadow .25s !important;
}
[data-testid="stFileUploader"]:hover {
  border-color: var(--blue) !important;
  box-shadow: 0 0 20px var(--blue-dim) !important;
}

/* ── Diagnosis badges ── */
.diag-badge {
  display: inline-flex; align-items: center; gap: .5rem;
  font-family: 'Space Grotesk', sans-serif;
  font-size: 1.6rem; font-weight: 700;
  padding: .5rem 1.1rem; border-radius: 50px;
}
.diag-normal    { background: rgba(74,222,128,.12); color: var(--green); border: 1px solid rgba(74,222,128,.3); }
.diag-parkinson { background: rgba(248,113,113,.12); color: var(--red);   border: 1px solid rgba(248,113,113,.3); }

/* ── Stat tiles ── */
.stat-tile {
  background: var(--bg2);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 1rem 1.2rem;
  text-align: center;
}
.stat-value {
  font-family: 'Space Grotesk', sans-serif;
  font-size: 1.9rem; font-weight: 700; color: var(--blue);
  line-height: 1.1; margin-bottom: .25rem;
}
.stat-label {
  font-family: 'IBM Plex Mono', monospace;
  font-size: .67rem; color: var(--text-2);
  letter-spacing: 1.5px; text-transform: uppercase;
}

/* ── Risk badge ── */
.risk-low      { background: rgba(74,222,128,.12); color: #4ade80; border: 1px solid rgba(74,222,128,.3); border-radius: 50px; padding: .2rem .8rem; font-size: .8rem; font-weight: 600; }
.risk-moderate { background: rgba(251,191,36,.12);  color: #fbbf24; border: 1px solid rgba(251,191,36,.3);  border-radius: 50px; padding: .2rem .8rem; font-size: .8rem; font-weight: 600; }
.risk-high     { background: rgba(248,113,113,.12); color: #f87171; border: 1px solid rgba(248,113,113,.3); border-radius: 50px; padding: .2rem .8rem; font-size: .8rem; font-weight: 600; }

/* ── Prob bar ── */
.prob-row { margin: .5rem 0; }
.prob-label { font-size: .78rem; color: var(--text-2); margin-bottom: .22rem; display: flex; justify-content: space-between; }
.prob-track { background: rgba(255,255,255,.06); border-radius: 6px; height: 8px; overflow: hidden; }
.prob-fill-n { height: 100%; border-radius: 6px; background: linear-gradient(90deg, #065f46, #4ade80); transition: width .8s cubic-bezier(.22,1,.36,1); }
.prob-fill-p { height: 100%; border-radius: 6px; background: linear-gradient(90deg, #7f1d1d, #f87171); transition: width .8s cubic-bezier(.22,1,.36,1); }

/* ── Image frame ── */
.img-frame {
  border: 1px solid var(--border);
  border-radius: var(--radius);
  overflow: hidden;
  background: #040a12;
  box-shadow: var(--shadow);
}
.img-caption {
  font-family: 'IBM Plex Mono', monospace;
  font-size: .67rem; color: var(--text-3);
  text-align: center; padding: .4rem 0 .2rem;
  letter-spacing: .8px;
}

/* ── Heatmap legend ── */
.hm-legend {
  display: flex; align-items: center; gap: .5rem;
  font-family: 'IBM Plex Mono', monospace;
  font-size: .65rem; color: var(--text-3);
  margin-top: .5rem;
}
.hm-bar { flex: 1; height: 7px; border-radius: 4px; background: linear-gradient(90deg,#00008b,#0000ff,#00ff00,#ffff00,#ff0000); }

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
  background: var(--bg1) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius) !important;
  padding: .3rem !important; gap: .2rem !important;
}
.stTabs [data-baseweb="tab"] {
  font-family: 'Space Grotesk', sans-serif !important;
  font-size: .86rem !important; font-weight: 600 !important;
  color: var(--text-2) !important; border-radius: var(--radius-sm) !important;
  padding: .55rem 1.1rem !important;
  transition: all .2s !important;
}
.stTabs [data-baseweb="tab"]:hover { color: var(--text) !important; }
.stTabs [aria-selected="true"] {
  background: var(--blue-dim) !important;
  color: var(--blue) !important;
  border: 1px solid rgba(59,130,246,.3) !important;
}
.stTabs [data-baseweb="tab-highlight"],
.stTabs [data-baseweb="tab-border"] { display: none !important; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #070c14, #0a1220) !important;
  border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * { font-family: 'DM Sans', sans-serif !important; }

/* ── Metrics ── */
[data-testid="stMetric"] {
  background: var(--bg2) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius) !important;
  padding: .9rem 1.1rem !important;
}
[data-testid="stMetricValue"] {
  font-family: 'Space Grotesk', sans-serif !important;
  font-size: 1.5rem !important; font-weight: 700 !important;
  color: var(--blue) !important;
}
[data-testid="stMetricLabel"] {
  color: var(--text-2) !important;
  font-family: 'IBM Plex Mono', monospace !important;
  font-size: .65rem !important; letter-spacing: 1px !important;
  text-transform: uppercase !important;
}

/* ── Progress ── */
.stProgress > div > div > div {
  background: linear-gradient(90deg, var(--blue), var(--teal)) !important;
  border-radius: 6px !important;
}

/* ── Table ── */
[data-testid="stDataFrame"] {
  border-radius: var(--radius) !important;
  border: 1px solid var(--border) !important;
  overflow: hidden !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--bg3); border-radius: 10px; }

/* ── HR ── */
hr {
  border: none !important; height: 1px !important;
  background: var(--border) !important;
  margin: 1.6rem 0 !important;
}

/* ── Alerts ── */
.stAlert { border-radius: var(--radius) !important; }

/* ── About cards ── */
.about-card {
  background: var(--bg2);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 1.6rem;
  text-align: center;
  height: 100%;
  transition: border-color .25s;
}
.about-card:hover { border-color: var(--border-hi); }

/* ── Team card ── */
.team-card {
  background: var(--bg2);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 1.2rem;
  text-align: center;
  transition: border-color .25s, transform .25s;
}
.team-card:hover { border-color: var(--border-hi); transform: translateY(-3px); }

/* ── Batch rows ── */
.batch-result {
  background: var(--bg2);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: .9rem 1.1rem;
  margin: .4rem 0;
}
.batch-normal    { border-left: 3px solid var(--green); }
.batch-parkinson { border-left: 3px solid var(--red); }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  HERO
# ══════════════════════════════════════════════════════════════════════════════
logo_src  = get_logo_b64("logo.png")
logo_html = (
    f'<img src="{logo_src}" style="width:72px;height:72px;object-fit:contain;'
    f'border-radius:50%;border:2px solid rgba(59,130,246,.4);"/>'
    if logo_src else
    '<div style="width:72px;height:72px;background:#0d1d35;border:2px solid rgba(59,130,246,.4);'
    'border-radius:50%;display:flex;align-items:center;justify-content:center;font-size:2rem;">🧠</div>'
)

st.markdown(
    '<div style="background:linear-gradient(180deg,#070d1a 0%,#080d14 100%);'
    'border-bottom:1px solid rgba(59,130,246,.14);padding:2.4rem 2rem 2rem;'
    'display:flex;flex-direction:column;align-items:center;gap:.8rem;">'
    + logo_html +
    '<div style="font-family:Space Grotesk,sans-serif;font-size:2.6rem;font-weight:700;'
    'color:#e2eaf4;letter-spacing:3px;line-height:1;">NEUROSCAN&nbsp;<span style="color:#3b82f6;">AI</span></div>'
    '<div style="font-family:IBM Plex Mono,monospace;font-size:.72rem;color:#4a6a8a;'
    'letter-spacing:3px;text-transform:uppercase;">Parkinson\'s Detection · Brain MRI · Deep Learning</div>'
    '<div style="display:flex;gap:1.4rem;margin-top:.4rem;flex-wrap:wrap;justify-content:center;">'
    '<span style="background:rgba(59,130,246,.1);border:1px solid rgba(59,130,246,.22);'
    'border-radius:50px;padding:.22rem .9rem;font-size:.75rem;color:#93c5fd;">🤖 ResNet50 + ViT</span>'
    '<span style="background:rgba(45,212,191,.08);border:1px solid rgba(45,212,191,.2);'
    'border-radius:50px;padding:.22rem .9rem;font-size:.75rem;color:#5eead4;">🎯 99.4% Accuracy</span>'
    '<span style="background:rgba(251,191,36,.08);border:1px solid rgba(251,191,36,.2);'
    'border-radius:50px;padding:.22rem .9rem;font-size:.75rem;color:#fcd34d;">🔥 Grad-CAM XAI</span>'
    '<span style="background:rgba(167,139,250,.08);border:1px solid rgba(167,139,250,.2);'
    'border-radius:50px;padding:.22rem .9rem;font-size:.75rem;color:#c4b5fd;">⚡ Batch Analysis</span>'
    '</div></div>',
    unsafe_allow_html=True,
)

# ── Model download ─────────────────────────────────────────────────────────────
download_model()

# ── Session state ─────────────────────────────────────────────────────────────
for k, v in [
    ("prediction_made",   False),
    ("patient_data",      {}),
    ("prediction_result", {}),
    ("batch_results",     []),
]:
    if k not in st.session_state:
        st.session_state[k] = v

# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### 🔬 NeuroScan AI")
    st.info(
        "Advanced deep-learning system for brain MRI analysis. "
        "Upload a scan to screen for Parkinson's Disease in seconds."
    )
    st.markdown("---")
    st.markdown("### 🤖 Model")
    st.success(
        "**Architecture:** ResNet50 + ViT\n\n"
        "**Classes:** Normal / Parkinson's\n\n"
        "**Validation Accuracy:** ~99.4%\n\n"
        "**XAI:** Grad-CAM\n\n"
        "**Status:** 🟢 Online"
    )
    st.markdown("---")
    st.markdown("### ✨ Features")
    st.markdown(
        "- 🧠 Single-scan analysis\n"
        "- 📦 Batch processing\n"
        "- 🔥 Grad-CAM heatmap\n"
        "- 📊 Risk scoring\n"
        "- 📄 PDF & CSV reports\n"
        "- 🩺 Doctor field support"
    )
    st.markdown("---")
    st.markdown("### ⚠️ Disclaimer")
    st.warning(
        "For **research/academic** purposes only. "
        "Not a substitute for clinical diagnosis by a licensed professional."
    )
    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1: st.metric("Precision", "100%")
    with c2: st.metric("Recall",    "98%")


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN TABS
# ══════════════════════════════════════════════════════════════════════════════
tab_scan, tab_batch, tab_about = st.tabs([
    "🧠  Single MRI Analysis",
    "📦  Batch Analysis",
    "🏫  About",
])


# ─────────────────────────────────────────────────────────────────────────────
#  TAB 1 — SINGLE MRI
# ─────────────────────────────────────────────────────────────────────────────
with tab_scan:
    col_form, col_upload = st.columns([1, 1], gap="large")

    # ── Patient form ─────────────────────────────────────────────────────────
    with col_form:
        st.markdown('<div class="sec-head">👤 Patient Information</div>', unsafe_allow_html=True)
        st.markdown('<div class="card">', unsafe_allow_html=True)

        patient_name = st.text_input("Full Name *", placeholder="Patient's full name")

        c1, c2 = st.columns(2)
        with c1: patient_age    = st.number_input("Age *", 0, 120, 45)
        with c2: patient_gender = st.selectbox("Gender *", ["Male", "Female", "Other"])

        c3, c4 = st.columns(2)
        with c3: patient_id  = st.text_input("Patient ID *", placeholder="P-2024-0001")
        with c4: scan_date   = st.date_input("Scan Date *", value=datetime.now())

        # NEW: Referring doctor field
        referring_doctor = st.text_input("Referring Doctor", placeholder="Dr. Name (optional)")

        medical_history = st.text_area("Medical History", placeholder="Relevant history (optional)", height=80)
        st.markdown('</div>', unsafe_allow_html=True)

    # ── Upload ────────────────────────────────────────────────────────────────
    with col_upload:
        st.markdown('<div class="sec-head">🖼️ MRI Image</div>', unsafe_allow_html=True)
        st.markdown('<div class="card">', unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "Upload Brain MRI Scan",
            type=["png", "jpg", "jpeg"],
            help="PNG / JPG / JPEG · any resolution",
        )
        if uploaded_file:
            image = Image.open(uploaded_file)
            st.markdown('<div class="img-frame">', unsafe_allow_html=True)
            st.image(image, caption="Uploaded MRI", use_column_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            # ── NEW: basic image stats ────────────────────────────────────────
            w, h = image.size
            st.markdown(
                f'<div style="font-family:IBM Plex Mono,monospace;font-size:.68rem;'
                f'color:#4a6a8a;margin-top:.4rem;text-align:center;">'
                f'{image.mode} · {w}×{h}px · {uploaded_file.size/1024:.1f} KB</div>',
                unsafe_allow_html=True,
            )
        else:
            st.info("⬆️ Upload a brain MRI scan to begin analysis.")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div class="sec-head">🔍 Analysis & Results</div>', unsafe_allow_html=True)

    btn_col, report_col = st.columns([1, 1], gap="large")

    with btn_col:
        if st.button("🚀 Analyze MRI Scan"):
            if not patient_name.strip():
                st.error("⚠️ Please enter the patient's full name.")
            elif not patient_id.strip():
                st.error("⚠️ Please enter a Patient ID.")
            elif uploaded_file is None:
                st.error("⚠️ Please upload a brain MRI scan.")
            else:
                with st.spinner("🧠 Running AI analysis…"):
                    st.session_state.patient_data = {
                        "name":            patient_name.strip(),
                        "age":             patient_age,
                        "gender":          patient_gender,
                        "patient_id":      patient_id.strip(),
                        "scan_date":       scan_date.strftime("%Y-%m-%d"),
                        "doctor":          referring_doctor.strip() or "—",
                        "medical_history": medical_history.strip(),
                    }
                    try:
                        model, device = load_model()
                        result        = predict_single(model, device, image)
                        st.session_state.prediction_result = result
                        st.session_state.prediction_made   = True
                        st.success("✅ Analysis complete!")
                        st.balloons()
                        st.rerun()
                    except Exception as exc:
                        st.error(f"❌ Error during analysis: {exc}")

    # ── RESULTS ───────────────────────────────────────────────────────────────
    if st.session_state.prediction_made:
        r         = st.session_state.prediction_result
        is_normal = r["class_idx"] == 0
        diag_cls  = "diag-normal" if is_normal else "diag-parkinson"
        risk_cls  = f"risk-{r['risk_level'].lower()}"

        # Diagnosis summary card
        st.markdown('<div class="card">', unsafe_allow_html=True)

        st.markdown("##### 📊 Diagnostic Summary")
        d1, d2, d3, d4 = st.columns(4)
        with d1:
            st.markdown(
                f'<div class="stat-tile"><div class="stat-label">Diagnosis</div>'
                f'<div style="margin-top:.4rem;"><span class="{diag_cls} diag-badge">'
                f'{"✅" if is_normal else "⚠️"} {r["prediction"]}</span></div></div>',
                unsafe_allow_html=True,
            )
        with d2:
            st.markdown(
                f'<div class="stat-tile"><div class="stat-label">Confidence</div>'
                f'<div class="stat-value">{r["confidence"]:.1f}%</div></div>',
                unsafe_allow_html=True,
            )
        with d3:
            st.markdown(
                f'<div class="stat-tile"><div class="stat-label">Risk Level</div>'
                f'<div style="margin-top:.5rem;"><span class="{risk_cls}">{r["risk_level"]} Risk</span></div></div>',
                unsafe_allow_html=True,
            )
        with d4:
            st.markdown(
                f'<div class="stat-tile"><div class="stat-label">Timestamp</div>'
                f'<div style="font-family:IBM Plex Mono,monospace;font-size:.78rem;'
                f'color:#8ba8c8;margin-top:.3rem;">{r["timestamp"]}</div></div>',
                unsafe_allow_html=True,
            )

        # Probability bars
        st.markdown('<br>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="prob-row">'
            f'<div class="prob-label"><span>✅ Normal</span><span>{r["normal_prob"]:.1f}%</span></div>'
            f'<div class="prob-track"><div class="prob-fill-n" style="width:{r["normal_prob"]:.1f}%"></div></div>'
            f'</div>'
            f'<div class="prob-row">'
            f'<div class="prob-label"><span>⚠️ Parkinson\'s</span><span>{r["parkinson_prob"]:.1f}%</span></div>'
            f'<div class="prob-track"><div class="prob-fill-p" style="width:{r["parkinson_prob"]:.1f}%"></div></div>'
            f'</div>',
            unsafe_allow_html=True,
        )
        st.markdown('</div>', unsafe_allow_html=True)

        # Grad-CAM
        st.markdown('<div class="sec-head">🔥 Grad-CAM Explainability</div>', unsafe_allow_html=True)
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown(
            '<p style="font-size:.84rem;color:#8ba8c8;margin-bottom:1rem;">'
            'Grad-CAM highlights the brain regions that drove the model\'s decision. '
            '<strong style="color:#3b82f6;">Warmer colours (red/yellow)</strong> indicate higher attention.</p>',
            unsafe_allow_html=True,
        )
        ic1, ic2, ic3 = st.columns(3)
        for col, img_obj, cap in [
            (ic1, r["image"],       "Original MRI"),
            (ic2, r["cam_heatmap"], "Attention Heatmap"),
            (ic3, r["cam_overlay"], "Grad-CAM Overlay"),
        ]:
            with col:
                st.markdown('<div class="img-frame">', unsafe_allow_html=True)
                st.image(img_obj, use_column_width=True)
                st.markdown(f'<div class="img-caption">{cap}</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

        st.markdown(
            '<div class="hm-legend"><span>Low</span><div class="hm-bar"></div><span>High</span></div>',
            unsafe_allow_html=True,
        )
        st.markdown('</div>', unsafe_allow_html=True)

    # ── Report button ─────────────────────────────────────────────────────────
    with report_col:
        if st.session_state.prediction_made:
            if st.button("📄 Generate PDF Report"):
                with st.spinner("📝 Building report…"):
                    try:
                        pdf_bytes = build_pdf(
                            st.session_state.patient_data,
                            st.session_state.prediction_result,
                        )
                        p = st.session_state.patient_data
                        fname = f"NeuroScan_{p['patient_id']}_{datetime.now().strftime('%Y%m%d')}.pdf"
                        st.download_button(
                            "⬇️ Download PDF Report",
                            data=pdf_bytes,
                            file_name=fname,
                            mime="application/pdf",
                        )
                        st.success("✅ PDF ready for download!")
                    except Exception as exc:
                        st.error(f"❌ PDF generation error: {exc}")
        else:
            st.info("📋 Run an analysis first to generate a report.")


# ─────────────────────────────────────────────────────────────────────────────
#  TAB 2 — BATCH
# ─────────────────────────────────────────────────────────────────────────────
with tab_batch:
    st.markdown('<div class="sec-head">📦 Batch MRI Analysis</div>', unsafe_allow_html=True)
    st.markdown(
        '<p style="color:#8ba8c8;font-size:.88rem;margin-bottom:1rem;">'
        'Upload multiple brain MRI scans at once. Each is independently analysed '
        'and results are summarised with a downloadable CSV report.</p>',
        unsafe_allow_html=True,
    )

    batch_files = st.file_uploader(
        "Upload Multiple Brain MRI Scans",
        type=["png", "jpg", "jpeg"],
        accept_multiple_files=True,
        key="batch_uploader",
    )

    if batch_files:
        st.info(f"📁 {len(batch_files)} file(s) ready for processing.")

        if st.button("🚀 Run Batch Analysis"):
            try:
                model, device = load_model()
            except Exception as exc:
                st.error(f"❌ Failed to load model: {exc}")
                st.stop()

            batch_results = []
            prog          = st.progress(0)
            status        = st.empty()

            for i, f in enumerate(batch_files):
                status.markdown(
                    f'<p style="font-family:IBM Plex Mono,monospace;font-size:.8rem;color:#3b82f6;">'
                    f'Processing {f.name} ({i+1}/{len(batch_files)})…</p>',
                    unsafe_allow_html=True,
                )
                try:
                    res = predict_single(model, device, Image.open(f))
                    res["filename"] = f.name
                    batch_results.append(res)
                except Exception as exc:
                    st.warning(f"⚠️ Skipped {f.name}: {exc}")
                prog.progress((i + 1) / len(batch_files))

            st.session_state.batch_results = batch_results
            status.empty()
            st.success(f"✅ Done! {len(batch_results)} scan(s) processed.")
            st.rerun()

    # ── Batch results ─────────────────────────────────────────────────────────
    if st.session_state.batch_results:
        results     = st.session_state.batch_results
        n_total     = len(results)
        n_normal    = sum(1 for r in results if r["class_idx"] == 0)
        n_park      = n_total - n_normal
        avg_conf    = np.mean([r["confidence"] for r in results])

        st.markdown('<div class="sec-head">📊 Summary</div>', unsafe_allow_html=True)
        m1, m2, m3, m4 = st.columns(4)
        with m1: st.metric("Total Scans",    n_total)
        with m2: st.metric("Normal",         n_normal)
        with m3: st.metric("Parkinson's",    n_park)
        with m4: st.metric("Avg Confidence", f"{avg_conf:.1f}%")

        st.markdown('<div class="sec-head">📈 Distribution</div>', unsafe_allow_html=True)
        ch_col, tb_col = st.columns([1, 1], gap="large")

        with ch_col:
            fig, ax = plt.subplots(figsize=(4.5, 3.8), facecolor="#0d1520")
            ax.set_facecolor("#0d1520")
            if n_normal > 0 or n_park > 0:
                wedges, texts, autotexts = ax.pie(
                    [n_normal, n_park],
                    labels=["Normal", "Parkinson's"],
                    autopct="%1.0f%%",
                    colors=["#4ade80", "#f87171"],
                    startangle=90,
                    wedgeprops=dict(edgecolor="#0d1520", linewidth=2.5),
                    textprops=dict(color="#e2eaf4", fontsize=11, fontfamily="DM Sans"),
                )
                for at in autotexts:
                    at.set_color("#0d1520"); at.set_fontweight("bold")
            ax.set_title("Scan Distribution", color="#8ba8c8", fontsize=11, pad=10)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        with tb_col:
            df = pd.DataFrame([{
                "File":          r["filename"],
                "Prediction":    r["prediction"],
                "Confidence":    f"{r['confidence']:.1f}%",
                "Normal %":      f"{r['normal_prob']:.1f}%",
                "Parkinson's %": f"{r['parkinson_prob']:.1f}%",
                "Risk":          r["risk_level"],
            } for r in results])
            st.dataframe(df, use_container_width=True, height=280)
            csv = df.to_csv(index=False)
            st.download_button(
                "⬇️ Download CSV Report",
                data=csv,
                file_name=f"batch_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
            )

        # Per-image cards
        st.markdown('<div class="sec-head">🖼️ Per-Image Results</div>', unsafe_allow_html=True)
        for r in results:
            is_n  = r["class_idx"] == 0
            bc    = "batch-normal" if is_n else "batch-parkinson"
            col   = "#4ade80" if is_n else "#f87171"
            icon  = "✅" if is_n else "⚠️"
            rc1, rc2, rc3, rc4 = st.columns([1, 2, 1, 1])
            with rc1:
                st.markdown('<div class="img-frame">', unsafe_allow_html=True)
                st.image(r["image"], use_column_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
            with rc2:
                st.markdown(
                    f'<p style="font-family:IBM Plex Mono,monospace;font-size:.72rem;'
                    f'color:#8ba8c8;margin-bottom:.2rem;">{r["filename"]}</p>'
                    f'<p style="font-family:Space Grotesk,sans-serif;font-size:1.4rem;'
                    f'font-weight:700;color:{col};margin:0;">{icon} {r["prediction"]}</p>'
                    f'<p style="font-family:IBM Plex Mono,monospace;font-size:.68rem;'
                    f'color:#4a6a8a;margin-top:.3rem;">{r["timestamp"]}</p>',
                    unsafe_allow_html=True,
                )
            with rc3:
                st.metric("Confidence", f"{r['confidence']:.1f}%")
                st.metric("Normal %",   f"{r['normal_prob']:.1f}%")
            with rc4:
                st.markdown('<div class="img-frame">', unsafe_allow_html=True)
                st.image(r["cam_overlay"], use_column_width=True)
                st.markdown('<div class="img-caption">Grad-CAM</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            st.markdown("<hr>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
#  TAB 3 — ABOUT
# ─────────────────────────────────────────────────────────────────────────────
with tab_about:
    college_logo = get_logo_b64("bvcr.jpg")
    clg_html     = (
        f'<img src="{college_logo}" style="width:90px;height:90px;object-fit:contain;margin-bottom:.8rem;"/>'
        if college_logo else '<div style="font-size:3rem;margin-bottom:.8rem;">🏛️</div>'
    )

    st.markdown(
        f'<div style="background:var(--bg2);border:1px solid var(--border-hi);border-radius:var(--radius);'
        f'padding:2.4rem;margin-bottom:1.4rem;text-align:center;">'
        f'{clg_html}'
        f'<div style="font-family:Space Grotesk,sans-serif;font-size:1.8rem;font-weight:700;'
        f'color:#e2eaf4;letter-spacing:2px;margin-bottom:.3rem;">BVC College of Engineering</div>'
        f'<div style="font-family:IBM Plex Mono,monospace;font-size:.72rem;color:#3b82f6;'
        f'letter-spacing:3px;text-transform:uppercase;margin-bottom:.8rem;">Autonomous</div>'
        f'<div style="font-size:.82rem;color:#8ba8c8;">Affiliated to JNTUK &nbsp;·&nbsp; AICTE Approved &nbsp;·&nbsp; NAAC A</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    ac1, ac2, ac3 = st.columns(3)
    for col, icon, title, body in [
        (ac1, "🎓", "About the College",
         "BVC College of Engineering, Palacharla is an <b style='color:#3b82f6;'>Autonomous</b> "
         "premier institution dedicated to excellence in engineering education."),
        (ac2, "🔬", "About the Project",
         "NeuroScan AI is a B.Tech final-year project for early Parkinson's screening from brain MRI, "
         "powered by a ResNet50 + ViT hybrid achieving 99.4% validation accuracy."),
        (ac3, "🤖", "Technology Stack",
         "<b style='color:#3b82f6;'>AI:</b> PyTorch · ResNet50 + ViT<br>"
         "<b style='color:#3b82f6;'>XAI:</b> Grad-CAM<br>"
         "<b style='color:#3b82f6;'>UI:</b> Streamlit<br>"
         "<b style='color:#3b82f6;'>Reports:</b> ReportLab · Pandas"),
    ]:
        with col:
            st.markdown(
                f'<div class="about-card">'
                f'<div style="font-size:2rem;margin-bottom:.6rem;">{icon}</div>'
                f'<div style="font-family:Space Grotesk,sans-serif;font-size:1rem;font-weight:600;'
                f'color:#93c5fd;margin-bottom:.6rem;">{title}</div>'
                f'<div style="font-size:.85rem;color:#8ba8c8;line-height:1.65;">{body}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    st.markdown('<div class="sec-head" style="margin-top:2rem;">👨‍💻 Project Team</div>', unsafe_allow_html=True)
    team = [
        {"roll": "236M5A0408", "name": "G Srinivasu",      "icon": "👨‍💻"},
        {"roll": "226M1A0460", "name": "S Anusha Devi",    "icon": "👩‍💻"},
        {"roll": "226M1A0473", "name": "V V Siva Vardhan", "icon": "👨‍💻"},
        {"roll": "236M5A0415", "name": "N L Sandeep",      "icon": "👨‍💻"},
    ]
    tcols = st.columns(4)
    for i, m in enumerate(team):
        with tcols[i]:
            st.markdown(
                f'<div class="team-card">'
                f'<div style="font-size:2rem;margin-bottom:.4rem;">{m["icon"]}</div>'
                f'<div style="font-family:Space Grotesk,sans-serif;font-size:.95rem;font-weight:600;'
                f'color:#93c5fd;margin-bottom:.2rem;">{m["name"]}</div>'
                f'<div style="font-family:IBM Plex Mono,monospace;font-size:.7rem;color:#4a6a8a;">{m["roll"]}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    st.markdown('<div class="sec-head" style="margin-top:2rem;">👨‍🏫 Project Guidance</div>', unsafe_allow_html=True)

    g1, g2, g3 = st.columns([2, 1, 1])
    with g1:
        st.markdown(
            '<div class="about-card" style="border-color:rgba(59,130,246,.35);">'
            '<div style="font-family:IBM Plex Mono,monospace;font-size:.65rem;color:#3b82f6;'
            'letter-spacing:2px;text-transform:uppercase;margin-bottom:.8rem;">⭐ Project Guide</div>'
            '<div style="font-size:2.4rem;margin-bottom:.5rem;">👨‍🏫</div>'
            '<div style="font-family:Space Grotesk,sans-serif;font-size:1.1rem;font-weight:700;'
            'color:#e2eaf4;margin-bottom:.6rem;">Ms. N P U V S N Pavan Kumar, M.Tech</div>'
            '<div style="display:flex;flex-wrap:wrap;gap:.4rem;justify-content:center;">'
            '<span style="background:rgba(59,130,246,.1);border:1px solid rgba(59,130,246,.25);'
            'border-radius:50px;padding:.18rem .7rem;font-size:.75rem;color:#93c5fd;">Assistant Professor</span>'
            '<span style="background:rgba(59,130,246,.1);border:1px solid rgba(59,130,246,.25);'
            'border-radius:50px;padding:.18rem .7rem;font-size:.75rem;color:#93c5fd;">Dept. of ECE</span>'
            '<span style="background:rgba(59,130,246,.1);border:1px solid rgba(59,130,246,.25);'
            'border-radius:50px;padding:.18rem .7rem;font-size:.75rem;color:#93c5fd;">Deputy CoE – III</span>'
            '</div></div>',
            unsafe_allow_html=True,
        )
    with g2:
        st.markdown(
            '<div class="about-card">'
            '<div style="font-family:IBM Plex Mono,monospace;font-size:.65rem;color:#2dd4bf;'
            'letter-spacing:2px;text-transform:uppercase;margin-bottom:.8rem;">📋 Coordinator</div>'
            '<div style="font-size:2rem;margin-bottom:.5rem;">📋</div>'
            '<div style="font-family:Space Grotesk,sans-serif;font-size:.95rem;font-weight:600;'
            'color:#e2eaf4;margin-bottom:.5rem;">Mr. K Anji Babu, M.Tech</div>'
            '<span style="background:rgba(45,212,191,.1);border:1px solid rgba(45,212,191,.25);'
            'border-radius:50px;padding:.18rem .7rem;font-size:.75rem;color:#5eead4;">Asst. Prof · ECE</span>'
            '</div>',
            unsafe_allow_html=True,
        )
    with g3:
        st.markdown(
            '<div class="about-card">'
            '<div style="font-family:IBM Plex Mono,monospace;font-size:.65rem;color:#fbbf24;'
            'letter-spacing:2px;text-transform:uppercase;margin-bottom:.8rem;">👨‍💼 HOD</div>'
            '<div style="font-size:2rem;margin-bottom:.5rem;">👨‍💼</div>'
            '<div style="font-family:Space Grotesk,sans-serif;font-size:.95rem;font-weight:600;'
            'color:#e2eaf4;margin-bottom:.5rem;">Dr. S A Vara Prasad, Ph.D</div>'
            '<div style="display:flex;flex-direction:column;gap:.3rem;align-items:center;">'
            '<span style="background:rgba(251,191,36,.1);border:1px solid rgba(251,191,36,.25);'
            'border-radius:50px;padding:.18rem .7rem;font-size:.75rem;color:#fcd34d;">Professor & HOD · ECE</span>'
            '<span style="background:rgba(251,191,36,.1);border:1px solid rgba(251,191,36,.25);'
            'border-radius:50px;padding:.18rem .7rem;font-size:.75rem;color:#fcd34d;">Chairman BoS</span>'
            '</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        '<div style="background:rgba(248,113,113,.07);border:1px solid rgba(248,113,113,.2);'
        'border-radius:var(--radius);padding:1rem 1.4rem;text-align:center;">'
        '<span style="font-size:.85rem;color:#fca5a5;">'
        '⚕️ This project is for <strong>academic and research purposes only</strong>. '
        'Always consult a qualified neurologist for any medical decisions.'
        '</span></div>',
        unsafe_allow_html=True,
    )


# ── FOOTER ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    '<div style="text-align:center;padding:1rem;">'
    '<div style="font-family:IBM Plex Mono,monospace;font-size:.65rem;color:#2a4a6a;'
    'letter-spacing:2px;text-transform:uppercase;margin-bottom:.3rem;">'
    'Research & Educational Use Only · Not for Clinical Diagnosis'
    '</div>'
    '<div style="font-size:.78rem;color:#3a6a8a;">'
    'NeuroScan AI · ResNet50 + ViT · Grad-CAM · PyTorch · Streamlit'
    '</div></div>',
    unsafe_allow_html=True,
)
