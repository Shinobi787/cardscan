# app.py
"""
Auto Business Card Scanner (EasyOCR) — Streamlit app

Features:
- Uses EasyOCR (no billing, no API keys)
- Uses streamlit camera_input for taking photos (works on Streamlit Cloud)
- Auto-deduplicates repeated frames (simple text-based check)
- Extracts Name, Company, Role, Phone, Email, Website using heuristics + regex
- Shows live table below and allows CSV download
- Saves to scans.csv for persistence across restarts

Notes:
- True continuous webcam streaming without user action requires WebRTC/TURN
  (not recommended on Streamlit Cloud). This app uses camera snapshots (the
  browser's "Take photo" button). It auto-adds new unique scans to the table.
"""

import streamlit as st
from PIL import Image
import easyocr
import numpy as np
import pandas as pd
import re
import io
from datetime import datetime

# -------------------- Config --------------------
st.set_page_config(page_title="Auto Business Card Scanner (EasyOCR)", layout="wide")
DATA_FILE = "scans.csv"

# -------------------- OCR Reader (cached) --------------------
@st.cache_resource
def get_reader():
    # 'en' only for speed; add more languages if you need
    return easyocr.Reader(['en'], gpu=False)

reader = get_reader()

# -------------------- Regex / Heuristics --------------------
PHONE_REGEX = re.compile(r"(\+?\d[\d\-\s\(\)]{7,}\d)")
EMAIL_REGEX = re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+")
WEBSITE_REGEX = re.compile(r"(https?://\S+|www\.\S+|\w+\.\w{2,})")

def extract_fields_from_text(text: str) -> dict:
    """
    Heuristic extraction. Not perfect but works well for most business cards.
    """
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    data = {"Name": "", "Company": "", "Role": "", "Phone": "", "Email": "", "Website": ""}

    if not lines:
        return data

    # Basic assumptions: first line = name, second = role/company, third = company/role
    data["Name"] = lines[0]
    if len(lines) >= 2:
        # Try to detect role keywords in second line
        second = lines[1]
        if re.search(r'\b(manager|director|founder|ceo|cto|engineer|designer|lead|head|officer|consultant|analyst|owner|president)\b', second, re.IGNORECASE):
            data["Role"] = second
            if len(lines) >= 3:
                data["Company"] = lines[2]
        else:
            data["Company"] = second
            if len(lines) >= 3:
                data["Role"] = lines[2]

    # Email
    emails = EMAIL_REGEX.findall(text)
    if emails:
        data["Email"] = emails[0]

    # Phone
    phones = PHONE_REGEX.findall(text)
    if phones:
        phone = phones[0]
        phone = re.sub(r"[^\d+]", "", phone)
        data["Phone"] = phone

    # Website
    webs = WEBSITE_REGEX.findall(text)
    if webs:
        # choose the most plausible
        for w in webs:
            if w.startswith("http") or w.startswith("www") or "." in w:
                data["Website"] = w
                break

    # Cleanup: remove found email/phone from name/company/role
    for k in ("Name", "Company", "Role"):
        if data[k]:
            data[k] = EMAIL_REGEX.sub("", data[k])
            data[k] = PHONE_REGEX.sub("", data[k])
            data[k] = data[k].strip()

    return data

# -------------------- Storage --------------------
def load_saved():
    try:
        df = pd.read_csv(DATA_FILE)
        return df
    except Exception:
        return pd.DataFrame(columns=["Name","Company","Role","Phone","Email","Website","Timestamp"])

def save_entry(entry: dict):
    df = load_saved()
    # append and save
    df = pd.concat([df, pd.DataFrame([entry])], ignore_index=True)
    df.to_csv(DATA_FILE, index=False)

# -------------------- Duplicate detection --------------------
def normalize_text_for_dup(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r'[^a-z0-9\s]', ' ', s)
    s = re.sub(r'\s+', ' ', s)
    return s

def is_duplicate(new_text: str, existing_texts: list, threshold_overlap=0.8) -> bool:
    """
    Simple Jaccard similarity on words to detect duplicates.
    """
    if not new_text:
        return True
    a = set(new_text.split())
    for t in existing_texts:
        b = set(t.split())
        if not a or not b:
            continue
        inter = len(a & b)
        union = len(a | b)
        j = inter / union if union else 0.0
        if j >= threshold_overlap:
            return True
    return False

# -------------------- UI --------------------
st.title("📇 Auto Business Card Scanner — EasyOCR (Free, Cloud-friendly)")
st.write("Hold the business card in front of your camera. Use the browser's 'Take photo' button. New unique scans will be appended to the table below automatically.")

col_left, col_right = st.columns([2,1])

with col_left:
    st.subheader("Camera")
    # camera_input returns a BytesIO-like object only after user clicks 'Take photo'
    img_bytes = st.camera_input("Click 'Take photo' to scan (works on desktop & mobile browsers)")

    # optional manual upload fallback
    st.write("Or upload an image:")
    uploaded_file = st.file_uploader("Upload image (.jpg/.png)", type=["jpg","jpeg","png"])

with col_right:
    st.subheader("Settings & Controls")
    dedup_threshold = st.slider("Duplicate similarity threshold", min_value=50, max_value=95, value=80, step=5, help="Higher = more strict (less duplicates). Expressed as percent.")
    dedup_threshold = dedup_threshold / 100.0
    auto_save = st.checkbox("Auto-save scans to scans.csv", value=True)
    show_raw = st.checkbox("Show raw OCR text when scanning", value=False)
    clear_session = st.button("Clear session table (file not deleted)")

# session storage init
if "detected" not in st.session_state:
    st.session_state.detected = []  # list of parsed dicts
if "normalized_texts" not in st.session_state:
    st.session_state.normalized_texts = []  # used for duplicate detection

# scanning logic (process whichever image is present; prefer camera snapshot)
image_obj = img_bytes if img_bytes is not None else uploaded_file

if image_obj is not None:
    # load PIL image
    try:
        pil_img = Image.open(image_obj).convert("RGB")
    except Exception as e:
        st.error(f"Can't open image: {e}")
        pil_img = None

    if pil_img is not None:
        # show preview
        st.image(pil_img, caption="Captured image", use_column_width=True)

        # Run OCR (EasyOCR)
        with st.spinner("Running OCR..."):
            np_img = np.array(pil_img)
            # reader.readtext returns list of text strings when detail=0
            try:
                ocr_lines = reader.readtext(np_img, detail=0, paragraph=True)
            except Exception as e:
                st.error(f"OCR failed: {e}")
                ocr_lines = []

            # join result
            ocr_text = "\n".join(ocr_lines).strip()

        if show_raw:
            st.subheader("Raw OCR text")
            st.text_area("raw", value=ocr_text, height=200)

        # Check dupe
        norm = normalize_text_for_dup(ocr_text)
        if not is_duplicate(norm, st.session_state.normalized_texts, threshold_overlap=dedup_threshold) and len(norm) > 10:
            parsed = extract_fields_from_text(ocr_text)
            parsed["Timestamp"] = datetime.utcnow().isoformat()
            # add to session table (newest first)
            st.session_state.detected.insert(0, parsed)
            st.session_state.normalized_texts.insert(0, norm)
            if auto_save:
                save_entry(parsed)
            st.success("New unique card scanned and saved.")
        else:
            st.info("Scan appears duplicate or empty — ignored.")

# allow clearing session
if clear_session:
    st.session_state.detected = []
    st.session_state.normalized_texts = []
    st.success("Session cleared (file not deleted).")

# show table (merge with file-saved entries)
saved_df = load_saved()
session_df = pd.DataFrame(st.session_state.detected)

# combine: session entries first, then saved (dedupe by exact Timestamp to avoid duplicates)
if not session_df.empty:
    combined = pd.concat([session_df, saved_df], ignore_index=True)
else:
    combined = saved_df

st.markdown("---")
st.subheader("📑 Scanned Cards")
if combined.empty:
    st.info("No scanned cards yet. Take a photo or upload an image.")
else:
    st.dataframe(combined.fillna(""), use_container_width=True)
    csv = combined.to_csv(index=False).encode("utf-8")
    st.download_button("Download CSV", data=csv, file_name="scans.csv", mime="text/csv")

st.markdown("---")
st.caption("Notes: For continuous hands-free stream you'll need WebRTC/TURN (not recommended on Streamlit Cloud). This app is optimized for Streamlit Cloud using snapshot captures.")
