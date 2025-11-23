import streamlit as st
from PIL import Image
from google.cloud import vision
import json
import io
import pandas as pd
import re
import time
from datetime import datetime

# --------------------- CONFIG ---------------------
st.set_page_config(page_title="Auto Business Card Scanner", layout="wide")
DATA_FILE = "scans.csv"

# --------------------- GOOGLE VISION ---------------------
def init_vision():
    try:
        key = st.secrets["google"]["gcp_key"]
        info = json.loads(key)
        client = vision.ImageAnnotatorClient.from_service_account_info(info)
        return client
    except Exception as e:
        st.error("Google Vision client not initialized. Check your Streamlit Secrets.")
        return None

client = init_vision()

# --------------------- OCR USING GOOGLE VISION ---------------------
def google_ocr(pil_image):
    buf = io.BytesIO()
    pil_image.convert("RGB").save(buf, format="JPEG")
    content = buf.getvalue()

    img = vision.Image(content=content)
    response = client.text_detection(image=img)

    if response.error.message:
        return ""

    if response.full_text_annotation:
        return response.full_text_annotation.text
    return ""

# --------------------- FIELD EXTRACTION ---------------------
PHONE_REGEX = re.compile(r"(\+?\d[\d\-\s\(\)]{7,}\d)")
EMAIL_REGEX = re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+")
WEBSITE_REGEX = re.compile(r"(https?://\S+|www\.\S+|\w+\.\w{2,})")

def extract_fields(text):
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    data = {"Name": "", "Company": "", "Role": "", "Phone": "", "Email": "", "Website": ""}

    if not lines:
        return data

    data["Name"] = lines[0]
    if len(lines) > 1:
        data["Company"] = lines[1]
    if len(lines) > 2:
        data["Role"] = lines[2]

    emails = EMAIL_REGEX.findall(text)
    if emails:
        data["Email"] = emails[0]

    phones = PHONE_REGEX.findall(text)
    if phones:
        p = re.sub(r"[^+0-9]", "", phones[0])
        data["Phone"] = p

    webs = WEBSITE_REGEX.findall(text)
    if webs:
        data["Website"] = webs[0]

    return data

# --------------------- LOAD/SAVE ---------------------
def load_data():
    try:
        return pd.read_csv(DATA_FILE)
    except:
        return pd.DataFrame(columns=["Name", "Company", "Role", "Phone", "Email", "Website", "Timestamp"])

def save_entry(entry):
    df = load_data()
    df.loc[len(df)] = entry
    df.to_csv(DATA_FILE, index=False)

# --------------------- MAIN UI ---------------------
st.title("📷 Auto Business Card Scanner — Google Vision (No Errors)")

st.write("💙 Hold the card in front of the camera. It will auto-scan every 2 seconds.")

placeholder_img = st.empty()
placeholder_status = st.empty()

if "last_scan" not in st.session_state:
    st.session_state.last_scan = ""

if "data" not in st.session_state:
    st.session_state.data = load_data()

# CAMERA AUTO SCAN LOOP
uploaded = st.camera_input("Show your card to the camera", key="camera")

if uploaded and client:
    image = Image.open(uploaded)
    placeholder_img.image(image, caption="Latest frame", use_column_width=True)

    text = google_ocr(image)

    if text and text != st.session_state.last_scan:
        placeholder_status.success("Scanned!")
        fields = extract_fields(text)
        fields["Timestamp"] = datetime.utcnow().isoformat()

        st.session_state.data.loc[len(st.session_state.data)] = fields
        save_entry(fields)

        st.session_state.last_scan = text

# --------------------- DISPLAY TABLE ---------------------
st.subheader("📑 Detected Cards")
st.dataframe(st.session_state.data, use_container_width=True)

csv = st.session_state.data.to_csv(index=False).encode("utf-8")
st.download_button("Download CSV", csv, "cards.csv", "text/csv")

