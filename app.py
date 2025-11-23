# app.py
"""
Streamlit Business Card Scanner
- Use camera_input (single image) or upload
- Preprocess image with OpenCV to improve OCR
- Extract: Name, Company, Role, Phone, Email, Website
- Let user edit before saving; stores to scans.csv and session
"""

import streamlit as st
from PIL import Image
import numpy as np
import cv2
import pytesseract
import re
import pandas as pd
import os
from datetime import datetime
from google.cloud import vision
import json

# ------------------ Configuration ------------------
st.set_page_config(page_title="Business Card Scanner", layout="wide")
DATA_FILE = "scans.csv"

# ------------------ Regex patterns ------------------
PHONE_REGEX = re.compile(r"(\+?\d[\d\-\s\(\)]{7,}\d)")
EMAIL_REGEX = re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+")
WEBSITE_REGEX = re.compile(r"(https?://\S+|www\.\S+|\w+\.\w{2,})")

# ------------------ Image preprocessing ------------------
def preprocess_image(pil_img):
    """
    Convert PIL image to OpenCV grayscale, denoise, apply CLAHE and adaptive threshold.
    Returns processed single-channel image ready for OCR.
    """
    img = np.array(pil_img.convert("RGB"))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    h, w = img.shape[:2]
    max_w = 1200
    if w > max_w:
        scale = max_w / w
        img = cv2.resize(img, (int(w * scale), int(h * scale)))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Denoise
    gray = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
    # Improve contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    # Adaptive threshold to remove background
    th = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY, 31, 15)
    return th

def image_to_text(pil_img):
    """Run preprocessing + Tesseract OCR and return plain text."""
    try:
        processed = preprocess_image(pil_img)
        # pytesseract can accept numpy array (grayscale)
        text = pytesseract.image_to_string(processed, lang='eng')
        return text
    except Exception as e:
        st.error(f"OCR error: {e}")
        return ""

# ------------------ Field extraction heuristics ------------------
def extract_fields(text):
    """
    Extract Name, Company, Role, Phone, Email, Website using heuristics.
    This is heuristic-based and will work for many cards but not all.
    """
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    data = {'Name': '', 'Company': '', 'Role': '', 'Phone': '', 'Email': '', 'Website': ''}

    # Emails
    emails = EMAIL_REGEX.findall(text)
    if emails:
        data['Email'] = emails[0]

    # Phones
    phones = PHONE_REGEX.findall(text)
    if phones:
        phone = max(phones, key=len)
        phone = re.sub(r"[^+0-9]", "", phone)
        data['Phone'] = phone

    # Websites
    webs = WEBSITE_REGEX.findall(text)
    if webs:
        # pick reasonable-looking entry
        for w in webs:
            if w.startswith("http") or w.startswith("www") or "." in w:
                data['Website'] = w
                break

    # Name/Company/Role heuristics using top lines
    if len(lines) >= 1:
        data['Name'] = lines[0]
    if len(lines) >= 2:
        second = lines[1]
        # company keywords
        if re.search(r'\b(company|co\.|pvt|private|ltd|inc|enterprises|solutions|technologies|tech|labs)\b', second, re.IGNORECASE):
            data['Company'] = second
            if len(lines) >= 3:
                data['Role'] = lines[2]
        else:
            # if second looks like a role/title
            if re.search(r'\b(manager|director|founder|ceo|coo|cto|engineer|designer|lead|head|officer|consultant|analyst|owner)\b', second, re.IGNORECASE):
                data['Role'] = second
                if len(lines) >= 3:
                    data['Company'] = lines[2]
            else:
                data['Company'] = second
                if len(lines) >= 3:
                    data['Role'] = lines[2]

    # fallback: detect uppercase lines likely company
    if not data['Company']:
        for l in lines[:5]:
            if l.isupper() and 2 < len(l.split()) <= 6:
                data['Company'] = l
                break

    # cleanup: strip phone/email residuals from textual fields
    for k in ['Name', 'Company', 'Role']:
        if data[k]:
            data[k] = re.sub(EMAIL_REGEX, '', data[k]).strip()
            data[k] = re.sub(PHONE_REGEX, '', data[k]).strip()

    return data

# ------------------ Persistence ------------------
def load_saved():
    if os.path.exists(DATA_FILE):
        try:
            return pd.read_csv(DATA_FILE)
        except Exception:
            return pd.DataFrame(columns=['Name','Company','Role','Phone','Email','Website','Timestamp'])
    else:
        return pd.DataFrame(columns=['Name','Company','Role','Phone','Email','Website','Timestamp'])

def save_entry(entry):
    df = load_saved()
    df = pd.concat([df, pd.DataFrame([entry])], ignore_index=True)
    df.to_csv(DATA_FILE, index=False)

# ------------------ UI ------------------
st.title("📇 Business Card Scanner (Camera or Upload)")

col1, col2 = st.columns([2,1])
with col1:
    st.header("Scan a card")
    camera_img = st.camera_input("Capture a single business card (camera)")
    uploaded_file = st.file_uploader("Or upload an image (png/jpg/jpeg)", type=['png','jpg','jpeg'])

with col2:
    st.header("Options")
    show_raw = st.checkbox("Show raw OCR text", value=False)
    auto_save = st.checkbox("Auto-save to scans.csv", value=True)
    st.markdown("---")
    st.markdown("**Tips:** good light, steady camera, card fills frame as much as possible.")

# decide which image to use
image_to_process = None
if camera_img is not None and uploaded_file is not None:
    st.info("Both camera capture and uploaded file present — using camera capture by default.")
if camera_img is not None:
    image_to_process = Image.open(camera_img)
elif uploaded_file is not None:
    image_to_process = Image.open(uploaded_file)

if image_to_process is not None:
    st.subheader("Preview")
    st.image(image_to_process, use_column_width=True)
    if st.button("Run OCR and Extract"):
        with st.spinner("Running OCR..."):
            text = image_to_text(image_to_process)
        if show_raw:
            st.subheader("Raw OCR")
            st.text_area("ocr", text, height=250)
        fields = extract_fields(text)
        fields['Timestamp'] = datetime.utcnow().isoformat()

        st.subheader("Check & edit extracted fields")
        name = st.text_input("Name", value=fields['Name'])
        company = st.text_input("Company", value=fields['Company'])
        role = st.text_input("Role", value=fields['Role'])
        phone = st.text_input("Phone", value=fields['Phone'])
        email = st.text_input("Email", value=fields['Email'])
        website = st.text_input("Website", value=fields['Website'])

        if st.button("Save to table"):
            entry = {
                'Name': name.strip(),
                'Company': company.strip(),
                'Role': role.strip(),
                'Phone': phone.strip(),
                'Email': email.strip(),
                'Website': website.strip(),
                'Timestamp': fields['Timestamp']
            }
            if 'cards' not in st.session_state:
                st.session_state.cards = []
            st.session_state.cards.append(entry)
            if auto_save:
                save_entry(entry)
            st.success("Saved entry.")

st.markdown("---")
st.subheader("Scanned Cards")

saved_df = load_saved()
session_df = pd.DataFrame(st.session_state.get('cards', [])) if 'cards' in st.session_state else pd.DataFrame()
display_df = pd.concat([session_df, saved_df], ignore_index=True) if not session_df.empty else saved_df

if display_df.empty:
    st.info("No scanned cards yet.")
else:
    st.dataframe(display_df[['Name','Company','Role','Phone','Email','Website','Timestamp']].fillna(''))
    csv = display_df.to_csv(index=False).encode('utf-8')
    st.download_button("Download CSV", csv, file_name="scans.csv", mime="text/csv")

st.markdown("---")
st.caption("App uses OpenCV + Tesseract OCR. For best accuracy, consider training OCR models or using a commercial OCR API.")

