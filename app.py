# app.py
"""
Auto Business Card Scanner — EasyOCR + improved preprocessing (NO OpenCV)
- No cv2, no Paddle, no ONNX, no API keys
- Preprocessing: resize, sharpen, denoise, autocontrast
- OCR: EasyOCR
- Smart extractor for Name, Company, Role, Phone, TollFree, Email, Website
- Auto-scan via st.camera_input (snapshot), plus "Test with sample" button using your uploaded file
"""

import streamlit as st
from PIL import Image, ImageFilter, ImageOps, ImageEnhance
import easyocr
import numpy as np
import pandas as pd
import re
import io
from datetime import datetime
import os

# ---------------- Config ----------------
st.set_page_config(page_title="Business Card Scanner (EasyOCR, No Crop)", layout="wide")
DATA_FILE = "scans.csv"
SAMPLE_PATH = "/mnt/data/07c1090d-a232-4e82-bb1f-16abd2b9ea93.png"  # your uploaded sample image path

# ----------------- OCR Reader -----------------
@st.cache_resource
def get_reader():
    # English only (fast). Add more languages in list if needed.
    return easyocr.Reader(['en'], gpu=False)

reader = get_reader()

# ----------------- Regex patterns -----------------
PHONE_REGEX = re.compile(r"(\+?\d[\d\-\s\(\)]{6,}\d)")
EMAIL_REGEX = re.compile(r"[a-zA-Z0-9_.+-]{2,}@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+")
WEBSITE_REGEX = re.compile(r"(https?://\S+|www\.\S+|\b[A-Za-z0-9-]+\.(com|in|io|net|org|co|biz|info|store)\b)")

# ----------------- Preprocessing helpers -----------------
def enhance_image_for_ocr(pil_img: Image.Image) -> Image.Image:
    """
    Improve readability for OCR using Pillow-only operations:
    - Resize up (2x) to increase DPI
    - Convert to RGB
    - Auto-contrast
    - Slight sharpening
    - Denoise via median filter
    - Increase contrast with ImageEnhance
    """
    img = pil_img.convert("RGB")
    # resize up for small camera images
    w, h = img.size
    scale = 1
    if max(w, h) < 1200:
        scale = int(1200 / max(w, h))  # scale to make max side ~1200
        img = img.resize((w * scale, h * scale), Image.LANCZOS)

    # auto-contrast
    img = ImageOps.autocontrast(img, cutoff=1)

    # denoise (median) and sharpen
    img = img.filter(ImageFilter.MedianFilter(size=3))
    img = img.filter(ImageFilter.SMOOTH)
    enhancer = ImageEnhance.Sharpness(img)
    img = enhancer.enhance(1.3)

    # slight brightness/contrast boost
    enhancer_c = ImageEnhance.Contrast(img)
    img = enhancer_c.enhance(1.15)

    return img

# ----------------- OCR wrapper -----------------
def ocr_easy(pil_img: Image.Image) -> str:
    np_img = np.array(pil_img)  # RGB
    try:
        texts = reader.readtext(np_img, detail=0, paragraph=True)
        text = "\n".join([t.strip() for t in texts if t.strip()])
        return text
    except Exception as e:
        st.error(f"EasyOCR failed: {e}")
        return ""

# ----------------- Improved extractor -----------------
ROLE_KEYWORDS = [
    "founder","co-founder","cofounder","director","chief","ceo","cto","cfo","coo","manager",
    "lead","vp","vice president","president","head","engineer","developer","designer",
    "analyst","consultant","owner","chairman","marketing","sales","product","operations",
    "hr","support","executive","md","proprietor","gm"
]
COMPANY_SUFFIX = [
    "pvt","private","limited","ltd","llp","inc","co","company","technologies",
    "solutions","studios","labs","enterprise","furnishings","industries","group","stores","pvt."
]

def normalize_line(s: str) -> str:
    return re.sub(r'\s+', ' ', s).strip()

def recover_email_from_fragment(s: str) -> str:
    s0 = s.strip()
    s0 = s0.replace('(at)', '@').replace('[at]', '@').replace(' at ', '@').replace(' AT ', '@')
    s0 = s0.replace('(dot)', '.').replace('[dot]', '.').replace(' dot ', '.')
    s0 = re.sub(r'\s+', '', s0)
    # If still missing '@' but has domain-like part, attempt to insert '@'
    if '@' not in s0:
        m = re.search(r'([A-Za-z0-9._%-]+)([A-Za-z0-9.-]+\.[A-Za-z]{2,})', s0)
        if m:
            return m.group(1) + "@" + m.group(2)
    return s0

def pick_name(lines):
    # prefer the first short alphabetic line that isn't a role/company
    for line in lines[:5]:
        plain = re.sub(r'[^A-Za-z\s]', ' ', line).strip()
        words = [w for w in plain.split() if len(w) > 1]
        if 1 < len(words) <= 4:
            lw = line.lower()
            if not any(k in lw for k in ROLE_KEYWORDS) and not any(s in lw for s in COMPANY_SUFFIX):
                return ' '.join([w.capitalize() for w in words])
    # fallback: first line title-cased
    if lines:
        return lines[0].title()
    return ""

def pick_role(lines):
    for line in lines[:6]:
        lw = line.lower()
        for k in ROLE_KEYWORDS:
            if k in lw:
                return normalize_line(line).title()
    return ""

def pick_company(lines):
    # check lower half first (company often near bottom)
    for line in reversed(lines[-6:]):
        lw = line.lower()
        if any(s in lw for s in COMPANY_SUFFIX):
            return normalize_line(line).title()
    # fallback: uppercase short line or second line
    for line in lines[:6]:
        if line.isupper() and 1 < len(line.split()) <= 6:
            return line.title()
    if len(lines) >= 2:
        return lines[1].title()
    return ""

def pick_phones(text_block):
    found = PHONE_REGEX.findall(text_block)
    cleaned = []
    for p in found:
        p2 = re.sub(r'[^0-9+]', '', p)
        if len(p2) >= 7:
            cleaned.append(p2)
    mobiles = []
    tollfree = []
    for p in cleaned:
        digits = re.sub(r'\D', '', p)
        if digits.startswith('1800') or digits.startswith('0800'):
            tollfree.append(p)
        if len(digits) == 10 or (len(digits) == 12 and digits.startswith('91')) or (len(digits) == 11 and digits.startswith('0')):
            mobiles.append(p)
    primary = mobiles[0] if mobiles else (cleaned[0] if cleaned else "")
    toll = tollfree[0] if tollfree else ""
    return primary, toll

def pick_email(text_block):
    emails = EMAIL_REGEX.findall(text_block)
    if emails:
        return emails[0]
    # try to salvage fragments
    frags = re.findall(r'[A-Za-z0-9._%+-]{2,}\s*(?:@|at|\(at\)|\[at\])\s*[A-Za-z0-9.-]{2,}', text_block)
    if frags:
        return recover_email_from_fragment(frags[0])
    # fallback: tokens with dot and letters
    tokens = re.findall(r'[A-Za-z0-9._%+-]{3,}\.[A-Za-z]{2,}', text_block)
    if tokens:
        return recover_email_from_fragment(tokens[0])
    return ""

def pick_website(text_block):
    webs = WEBSITE_REGEX.findall(text_block)
    if webs:
        return webs[0][0] if isinstance(webs[0], tuple) else webs[0]
    m = re.search(r'([A-Za-z0-9-]+\.(com|in|io|net|org|co|biz|info|store))', text_block, re.IGNORECASE)
    if m:
        return m.group(0)
    return ""

def extract_fields_from_text(text_block: str) -> dict:
    lines = [normalize_line(l) for l in text_block.splitlines() if l.strip()]
    name = pick_name(lines)
    role = pick_role(lines)
    company = pick_company(lines)
    phone, toll = pick_phones(text_block)
    email = pick_email(text_block)
    website = pick_website(text_block)
    return {
        "Name": name,
        "Company": company,
        "Role": role,
        "Phone": phone,
        "TollFree": toll,
        "Email": email,
        "Website": website
    }

# ----------------- Storage -----------------
def load_saved():
    try:
        return pd.read_csv(DATA_FILE)
    except Exception:
        return pd.DataFrame(columns=["Name","Company","Role","Phone","TollFree","Email","Website","Timestamp"])

def save_entry(entry: dict):
    df = load_saved()
    df = pd.concat([df, pd.DataFrame([entry])], ignore_index=True)
    df.to_csv(DATA_FILE, index=False)

# ----------------- UI -----------------
st.title("📇 Business Card Scanner — Free & Robust (No OpenCV)")

col1, col2 = st.columns([2,1])
with col1:
    st.header("Camera / Upload")
    cam = st.camera_input("Take photo (browser control)")
    upload = st.file_uploader("Or upload an image", type=["jpg","jpeg","png"])
    st.markdown("**Quick test with your uploaded sample image:**")
    if st.button("Test with sample image (your uploaded file)"):
        if os.path.exists(SAMPLE_PATH):
            upload = SAMPLE_PATH
        else:
            st.error("Sample file not found on server.")

with col2:
    st.header("Options")
    auto_save = st.checkbox("Auto-save unique scans to scans.csv", value=True)
    show_raw = st.checkbox("Show raw OCR text", value=False)
    dedup_percent = st.slider("Duplicate similarity threshold (%)", 50, 95, 80)
    clear = st.button("Clear session (not file)")

if "seen_norm" not in st.session_state:
    st.session_state.seen_norm = []
if "detected" not in st.session_state:
    st.session_state.detected = []

# Pick input image (cam takes precedence)
input_img = None
if cam:
    try:
        input_img = Image.open(cam).convert("RGB")
    except Exception:
        input_img = None
elif upload:
    try:
        if isinstance(upload, str) and os.path.exists(upload):
            input_img = Image.open(upload).convert("RGB")
        else:
            input_img = Image.open(upload).convert("RGB")
    except Exception:
        input_img = None

if input_img is not None:
    st.image(input_img, caption="Captured image", use_column_width=True)
    # preprocess
    pre = enhance_image_for_ocr(input_img)
    # show preprocessed if chosen
    if show_raw:
        st.subheader("Preprocessed image used for OCR")
        st.image(pre, use_column_width=True)

    with st.spinner("Running OCR..."):
        ocr_text = ocr_easy(pre)

    if not ocr_text.strip():
        st.warning("No text detected. Try improving lighting or centering card.")
    else:
        if show_raw:
            st.subheader("Raw OCR output")
            st.text_area("raw", value=ocr_text, height=220)

        parsed = extract_fields_from_text(ocr_text)
        parsed["Timestamp"] = datetime.utcnow().isoformat()

        # dedupe by normalized block
        norm = re.sub(r'[^a-z0-9]', '', ocr_text.lower())
        # threshold not strict here; use exact-match logic with seen set
        if norm and norm not in st.session_state.seen_norm:
            st.session_state.seen_norm.insert(0, norm)
            st.session_state.detected.insert(0, parsed)
            if auto_save:
                save_entry(parsed)
            st.success("New card scanned and saved.")
        else:
            st.info("Duplicate or empty scan ignored.")

# clear session
if clear:
    st.session_state.detected = []
    st.session_state.seen_norm = []
    st.success("Session cleared.")

# show combined table
st.markdown("---")
st.subheader("Scanned Cards (session + saved)")
session_df = pd.DataFrame(st.session_state.detected)
saved_df = load_saved()
if not session_df.empty:
    combined = pd.concat([session_df, saved_df], ignore_index=True)
else:
    combined = saved_df

if combined.empty:
    st.info("No scans yet.")
else:
    st.dataframe(combined.fillna(""), use_container_width=True)
    csv = combined.to_csv(index=False).encode("utf-8")
    st.download_button("Download CSV", csv, "scans.csv", "text/csv")

st.caption("This version uses EasyOCR + robust preprocessing and extraction heuristics. No OpenCV, no external keys, runs on Streamlit Cloud.")
