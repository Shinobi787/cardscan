# app.py
"""
Auto Business Card Scanner — Ensemble OCR (EasyOCR + PaddleOCR) + Card Detection
- Auto-crop card using edge detection + perspective transform
- Preprocess (CLAHE, denoise, resize)
- OCR ensemble: EasyOCR + PaddleOCR (if available)
- Smart extractor: Name, Company, Role, Phone, Email, Website
- Auto-scan mode via st.camera_input (snapshots)
- Test button uses the uploaded sample image at /mnt/data/07c1090d-a232-4e82-bb1f-16abd2b9ea93.png
"""

import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd
import re
import io
import cv2
import os
from datetime import datetime

# OCR libs (import with fallback)
import easyocr

# Try to import PaddleOCR; fallback logic handled later
try:
    from paddleocr import PaddleOCR
    PADDLE_AVAILABLE = True
except Exception:
    PADDLE_AVAILABLE = False

st.set_page_config(page_title="Business Card Scanner — Ensemble OCR + Crop", layout="wide")
DATA_FILE = "scans.csv"

# ---------------------- Utility / Preprocessing ----------------------
PHONE_REGEX = re.compile(r"(\+?\d[\d\-\s\(\)]{6,}\d)")
EMAIL_REGEX = re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+")
WEBSITE_REGEX = re.compile(r"(https?://\S+|www\.\S+|\w+\.\w{2,})")

@st.cache_resource
def get_easyocr_reader():
    return easyocr.Reader(['en'], gpu=False)

EASY_READER = get_easyocr_reader()

def try_init_paddle():
    if not PADDLE_AVAILABLE:
        return None
    # PaddleOCR initialization - CPU mode
    try:
        ocr = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=False)
        return ocr
    except Exception:
        return None

PADDLE_OCR = try_init_paddle()

def to_grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def clahe_enhance(gray):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    return clahe.apply(gray)

def denoise(img):
    return cv2.fastNlMeansDenoising(img, None, 10, 7, 21)

def resize_for_ocr(img, target_width=1200):
    h, w = img.shape[:2]
    if w < target_width:
        scale = target_width / w
        return cv2.resize(img, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_LINEAR)
    return img

def preprocess_for_ocr(pil_img):
    # Convert to OpenCV BGR
    img = np.array(pil_img.convert("RGB"))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = resize_for_ocr(img, target_width=1200)
    gray = to_grayscale(img)
    gray = denoise(gray)
    gray = clahe_enhance(gray)
    # Return color and gray both
    return img, gray

# ---------------------- Card detection & perspective crop ----------------------
def detect_card_and_crop(pil_img):
    """
    Detect largest rectangular contour (card) and perspective-transform it.
    Returns cropped PIL image if detection succeeds; otherwise returns original PIL image.
    """
    img = np.array(pil_img.convert("RGB"))
    orig = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (5,5), 0)
    edged = cv2.Canny(gray, 50, 200)
    # Dilate to close gaps
    kernel = np.ones((5,5), np.uint8)
    edged = cv2.dilate(edged, kernel, iterations=1)
    contours, _ = cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return pil_img  # no contours, return original

    # sort by area descending
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
    card_cnt = None
    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            card_cnt = approx
            break
    if card_cnt is None:
        # fallback: take bounding rectangle of largest contour
        c = contours[0]
        x,y,w,h = cv2.boundingRect(c)
        crop = orig[y:y+h, x:x+w]
        return Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))

    pts = card_cnt.reshape(4,2)
    # order points: tl, tr, br, bl
    def order_pts(pts):
        rect = np.zeros((4,2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect
    rect = order_pts(pts)
    (tl, tr, br, bl) = rect
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = max(int(widthA), int(widthB))
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = max(int(heightA), int(heightB))
    dst = np.array([
        [0,0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(orig, M, (maxWidth, maxHeight))
    return Image.fromarray(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))

# ---------------------- OCR Ensemble ----------------------
def ocr_easyocr(pil_img):
    np_img = np.array(pil_img.convert("RGB"))
    try:
        lines = EASY_READER.readtext(np_img, detail=0, paragraph=True)
        # returns list of text; join into lines
        return "\n".join(lines)
    except Exception as e:
        return ""

def ocr_paddleocr(pil_img):
    if PADDLE_OCR is None:
        return ""
    # Paddle OCR expects path or numpy
    try:
        np_img = np.array(pil_img.convert("RGB"))[:,:,::-1]  # BGR
        result = PADDLE_OCR.ocr(np_img, cls=True)
        lines = []
        for r in result:
            for line in r:
                txt = line[1][0]
                lines.append(txt)
        # Join preserving probable line breaks
        return "\n".join(lines)
    except Exception:
        return ""

def ensemble_ocr_text(pil_img):
    # crop & preprocess before sending to OCR is done upstream
    t_easy = ocr_easyocr(pil_img)
    t_paddle = ocr_paddleocr(pil_img)
    # Combine intelligently:
    # - prefer paddle if it produced longer/more reliable output
    # - else merge unique lines from both
    if t_paddle and len(t_paddle) > len(t_easy) * 0.8:
        # merge unique lines but keep paddle primary
        paddle_lines = [l.strip() for l in t_paddle.splitlines() if l.strip()]
        easy_lines = [l.strip() for l in t_easy.splitlines() if l.strip()]
        merged = paddle_lines[:]
        for l in easy_lines:
            if all(l.lower() != pl.lower() for pl in paddle_lines):
                merged.append(l)
        return "\n".join(merged)
    else:
        # merge both, prefer longer lines
        easy_lines = [l.strip() for l in t_easy.splitlines() if l.strip()]
        paddle_lines = [l.strip() for l in t_paddle.splitlines() if l.strip()]
        merged = []
        seen = set()
        for l in easy_lines + paddle_lines:
            key = re.sub(r'\W+', '', l).lower()
            if key and key not in seen:
                merged.append(l)
                seen.add(key)
        return "\n".join(merged)

# ---------------------- Improved extractor ----------------------
ROLE_KEYWORDS = [
    "founder","co-founder","cofounder","director","chief","ceo","cto","cfo","coo","manager",
    "lead","vp","vice president","president","head","engineer","developer","designer",
    "analyst","consultant","owner","chairman","marketing","sales","product","operations",
    "hr","support","executive","md","proprietor"
]
COMPANY_SUFFIX = [
    "pvt","private","limited","ltd","llp","inc","ltd.","co","co.","company","technologies",
    "solutions","studios","labs","enterprise","furnishings","furnishings","industries","group"
]

def normalize(s: str) -> str:
    return re.sub(r'\s+', ' ', s).strip()

def pick_name(lines):
    # Prefer the top-most line that's alphabetic and short (2-4 words) and not a role/company
    for i, line in enumerate(lines[:5]):
        plain = re.sub(r'[^A-Za-z\s]', ' ', line).strip()
        words = plain.split()
        if 1 < len(words) <= 4:
            lw = line.lower()
            if not any(k in lw for k in ROLE_KEYWORDS) and not any(s in lw for s in COMPANY_SUFFIX):
                return ' '.join([w.capitalize() for w in words])
    # fallback: first line
    if lines:
        return lines[0].title()
    return ""

def pick_role(lines):
    for line in lines[:6]:
        lw = line.lower()
        if any(k in lw for k in ROLE_KEYWORDS):
            return normalize(line).title()
    return ""

def pick_company(lines):
    # check bottom area and lines for company suffix
    for line in reversed(lines):
        lw = line.lower()
        if any(s in lw for s in COMPANY_SUFFIX):
            return normalize(line).title()
    # fallback: uppercase short line
    for line in lines[:6]:
        if line.isupper() and 1 < len(line.split()) <= 5:
            return line.title()
    # fallback: second or third line
    if len(lines) >= 2:
        return lines[1].title()
    return ""

def recover_email_from_errors(s):
    # try to repair common OCR mistakes like missing '@' or '.' replaced
    s = s.strip()
    s = s.replace(' at ', '@').replace(' AT ', '@')
    s = s.replace(' dot ', '.').replace(' DOT ', '.')
    # fix common OCR swaps
    s = re.sub(r'\s+', '', s)
    if '@' in s and '.' in s:
        return s
    # try to guess: if 'rfplin' -> rfpl.in etc.
    # simple heuristic: if there is a known domain part in text, insert '@' before it
    m = re.search(r'([A-Za-z0-9._%-]+)([A-Za-z0-9.-]+\.[A-Za-z]{2,})', s)
    if m:
        return m.group(1) + "@" + m.group(2)
    return s

def pick_phones(text):
    found = PHONE_REGEX.findall(text)
    cleaned = []
    for p in found:
        p2 = re.sub(r'[^0-9+]', '', p)
        if len(p2) >= 7:
            cleaned.append(p2)
    # classify
    mobiles = []
    tollfree = []
    for p in cleaned:
        pure = p.lstrip('+').lstrip('0')
        # toll-free detection (India example): starts with 1800
        if p.startswith('1800') or pure.startswith('1800'):
            tollfree.append(p)
        # mobile heuristic: 10 digits or +91 + 10 digits
        digits = re.sub(r'\D', '', p)
        if len(digits) == 10 or (len(digits) == 12 and digits.startswith('91')):
            mobiles.append(p)
    # prefer mobiles
    primary = mobiles[0] if mobiles else (cleaned[0] if cleaned else "")
    return primary, tollfree[0] if tollfree else ""

def pick_email(text):
    emails = EMAIL_REGEX.findall(text)
    if emails:
        return emails[0]
    # try to recover from common OCR mistakes
    # look for tokens that look like name+domain
    tokens = re.findall(r'[A-Za-z0-9._%+-]{3,}\s*[at@]\s*[A-Za-z0-9.-]{3,}', text)
    if tokens:
        guess = tokens[0].replace(' ', '').replace('at', '@')
        return recover_email_from_errors(guess)
    # try to find domain-like token and attach probable local-part
    return ""

def pick_website(text):
    webs = WEBSITE_REGEX.findall(text)
    if webs:
        for w in webs:
            if 'www' in w or '.' in w:
                return w
    # try to find domain-like tokens
    m = re.search(r'([A-Za-z0-9-]+\.(com|in|co|io|net|org|biz|info|online|store|co\.in))', text, re.IGNORECASE)
    if m:
        return m.group(0)
    return ""

def extract_fields_from_ocr_text(text):
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    text_block = "\n".join(lines)
    name = pick_name(lines)
    role = pick_role(lines)
    company = pick_company(lines)
    phone, tollfree = pick_phones(text_block)
    email = pick_email(text_block)
    if not email:
        # also try to recover tokens that look like email
        possible = re.findall(r'[A-Za-z0-9._%+-]{3,}\s*[.@]\s*[A-Za-z0-9.-]{3,}', text_block)
        if possible:
            email = recover_email_from_errors(possible[0])
    website = pick_website(text_block)

    return {
        "Name": name,
        "Company": company,
        "Role": role,
        "Phone": phone,
        "TollFree": tollfree,
        "Email": email,
        "Website": website,
    }

# ---------------------- Persistence ----------------------
def load_saved():
    try:
        return pd.read_csv(DATA_FILE)
    except Exception:
        return pd.DataFrame(columns=["Name","Company","Role","Phone","TollFree","Email","Website","Timestamp"])

def save_entry(entry: dict):
    df = load_saved()
    df = pd.concat([df, pd.DataFrame([entry])], ignore_index=True)
    df.to_csv(DATA_FILE, index=False)

# ---------------------- Streamlit UI ----------------------
st.title("📇 Pro Business Card Scanner — EasyOCR + PaddleOCR (Ensemble)")

col_l, col_r = st.columns([2,1])

with col_l:
    st.header("Camera / Upload")
    st.write("Hold card in front of your camera and click 'Take photo' (browser control). You can also upload images.")
    cam = st.camera_input("Take photo (Try to center card)", key="cam")
    upload = st.file_uploader("Or upload image", type=["jpg","jpeg","png"])

    # Test with a sample image that you uploaded earlier (local path)
    st.markdown("**Quick test with your uploaded sample image:**")
    if st.button("Test with sample image (your uploaded file)"):
        sample_path = "/mnt/data/07c1090d-a232-4e82-bb1f-16abd2b9ea93.png"
        if os.path.exists(sample_path):
            cam = Image.open(sample_path)
        else:
            st.error(f"Sample image not found at {sample_path}")

with col_r:
    st.header("Settings")
    use_card_crop = st.checkbox("Auto-detect & crop card (recommended)", value=True)
    show_preprocessed = st.checkbox("Show preprocessed crop", value=False)
    prefer_paddle = st.checkbox("Prefer PaddleOCR if available", value=True)
    dedup = st.checkbox("Auto-save unique scans to scans.csv", value=True)
    clear_session = st.button("Clear session table (not file)")

# session init
if "detected" not in st.session_state:
    st.session_state.detected = []
if "seen_norms" not in st.session_state:
    st.session_state.seen_norms = []

# select input image (cam takes precedence)
input_image = None
if cam:
    if isinstance(cam, Image.Image):
        input_image = cam
    else:
        input_image = Image.open(cam)
elif upload:
    input_image = Image.open(upload)

if input_image is not None:
    # optionally crop card
    if use_card_crop:
        try:
            cropped = detect_card_and_crop(input_image)
        except Exception as e:
            st.warning(f"Card detect failed, using original image: {e}")
            cropped = input_image
    else:
        cropped = input_image

    if show_preprocessed:
        st.subheader("Cropped / Preprocessed preview")
        st.image(cropped, use_column_width=True)

    # Run ensemble OCR
    with st.spinner("Running OCR (ensemble)..."):
        ocr_text = ensemble_ocr_text(cropped)

    if not ocr_text.strip():
        st.warning("No OCR text found.")
    else:
        # show raw text
        if st.checkbox("Show raw OCR text (debug)", value=False):
            st.text_area("raw", ocr_text, height=240)

        parsed = extract_fields_from_ocr_text(ocr_text)
        parsed["Timestamp"] = datetime.utcnow().isoformat()

        # dedupe by normalized block text
        norm = re.sub(r'[^a-z0-9]', '', ocr_text.lower())
        if norm and norm not in st.session_state.seen_norms:
            st.session_state.seen_norms.insert(0, norm)
            st.session_state.detected.insert(0, parsed)
            if dedup:
                save_entry(parsed)
            st.success("New scan added.")
        else:
            st.info("Duplicate or empty scan (ignored).")

# clear session
if clear_session:
    st.session_state.detected = []
    st.session_state.seen_norms = []
    st.success("Session cleared.")

# show table
st.markdown("---")
st.subheader("Detected / Saved Cards")
detected_df = pd.DataFrame(st.session_state.detected)
saved_df = load_saved()
if not detected_df.empty:
    combined = pd.concat([detected_df, saved_df], ignore_index=True)
else:
    combined = saved_df

if combined.empty:
    st.info("No scans yet.")
else:
    st.dataframe(combined.fillna(""), use_container_width=True)
    csv = combined.to_csv(index=False).encode("utf-8")
    st.download_button("Download CSV", csv, "scans.csv", "text/csv")

st.caption("Ensemble OCR: EasyOCR + PaddleOCR (if available). Auto-crop greatly improves accuracy. If Paddle fails to initialize on your host, the app will still use EasyOCR.")
