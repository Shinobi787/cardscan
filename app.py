# app.py
"""
Auto Business Card Scanner — Streamlit + Google Vision + streamlit-webrtc
Auto-scans cards from webcam, runs Google Vision OCR automatically (every N frames),
parses fields (Name, Company, Role, Phone, Email, Website) and appends to a live table.
Saves entries to scans.csv if auto-save enabled.
"""

import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration
from google.cloud import vision
import io
import json
import threading
import pandas as pd
import re
import numpy as np
import cv2
from PIL import Image
from datetime import datetime
import os
import base64

# ---------------- Config ----------------
st.set_page_config(page_title="Auto Business Card Scanner", layout="wide")
DATA_FILE = "scans.csv"

# How frequently to run OCR (process every N-th frame)
OCR_EVERY_N_FRAMES = 15  # ~ once every 0.5-1s depending on camera FPS

# Minimal characters in OCR full text to be considered valid
MIN_OCR_TEXT_LEN = 20

# Deduplication threshold (if same text seen recently, skip)
DUPLICATE_SIMILARITY_THRESHOLD = 0.9

# ---------------- Regex ----------------
PHONE_REGEX = re.compile(r"(\+?\d[\d\-\s\(\)]{7,}\d)")
EMAIL_REGEX = re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+")
WEBSITE_REGEX = re.compile(r"(https?://\S+|www\.\S+|\w+\.\w{2,})")

# ---------------- Google Vision init ----------------
def init_vision_client():
    """
    Initialize the Google Vision client from Streamlit secrets.
    Put your service account JSON into Streamlit secrets under:
    [google]
    gcp_key = \"\"\"<full JSON here>\"\"\"
    """
    try:
        gcp_key = st.secrets["google"]["gcp_key"]
        info = json.loads(gcp_key)
        client = vision.ImageAnnotatorClient.from_service_account_info(info)
        return client
    except Exception as e:
        st.error("Google Vision client not initialized. Add your service account JSON to Streamlit secrets "
                 "under [google] gcp_key. See README for exactly how.")
        st.stop()

client = init_vision_client()

# ---------------- OCR helper ----------------
def google_ocr_from_pil(pil_image):
    """
    Send PIL image bytes to Google Vision and return full text (string).
    """
    buf = io.BytesIO()
    pil_image.convert("RGB").save(buf, format="JPEG")
    content = buf.getvalue()
    image = vision.Image(content=content)
    response = client.text_detection(image=image)
    if response.error and response.error.message:
        # Show error message
        st.warning(f"Vision API error: {response.error.message}")
        return ""
    text = ""
    if response.full_text_annotation and response.full_text_annotation.text:
        text = response.full_text_annotation.text
    return text

# ---------------- Field extraction ----------------
def extract_fields_from_text(text):
    """
    Heuristic extraction of Name, Company, Role, Phone, Email, Website from OCR text.
    Returns dict with keys: Name, Company, Role, Phone, Email, Website
    """
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    data = {'Name': '', 'Company': '', 'Role': '', 'Phone': '', 'Email': '', 'Website': ''}
    if not lines:
        return data

    # Email
    emails = EMAIL_REGEX.findall(text)
    if emails:
        data['Email'] = emails[0]

    # Phone
    phones = PHONE_REGEX.findall(text)
    if phones:
        phone = max(phones, key=len)
        phone = re.sub(r"[^+0-9]", "", phone)
        data['Phone'] = phone

    # Website
    webs = WEBSITE_REGEX.findall(text)
    if webs:
        for w in webs:
            if w.startswith("http") or w.startswith("www") or '.' in w:
                data['Website'] = w
                break

    # Basic heuristics: first line likely name, second might be role/company
    data['Name'] = lines[0]
    if len(lines) >= 2:
        second = lines[1]
        if re.search(r'\b(company|co\.|pvt|private|ltd|inc|enterprises|solutions|technologies|tech|labs)\b', second, re.IGNORECASE):
            data['Company'] = second
            if len(lines) >= 3:
                data['Role'] = lines[2]
        elif re.search(r'\b(manager|director|founder|ceo|coo|cto|engineer|designer|lead|head|officer|consultant|analyst|owner)\b', second, re.IGNORECASE):
            data['Role'] = second
            if len(lines) >= 3:
                data['Company'] = lines[2]
        else:
            data['Company'] = second
            if len(lines) >= 3:
                data['Role'] = lines[2]

    # fallback: uppercase likely company
    if not data['Company']:
        for l in lines[:5]:
            if l.isupper() and 2 < len(l.split()) <= 6:
                data['Company'] = l
                break

    # Clean name/company/role from stray phone/email fragments
    for k in ['Name', 'Company', 'Role']:
        if data[k]:
            data[k] = re.sub(EMAIL_REGEX, '', data[k]).strip()
            data[k] = re.sub(PHONE_REGEX, '', data[k]).strip()

    return data

# ---------------- Persistence ----------------
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

# ---------------- Global results store (thread-safe) ----------------
scan_results = []
scan_lock = threading.Lock()
last_text = {"value": ""}

def is_duplicate_text(new_text, old_text):
    """
    Simple duplicate check: exact or high overlap.
    We'll just check normalized similarity by set-of-words Jaccard.
    """
    if not new_text or not old_text:
        return False
    a = set(w.lower() for w in re.findall(r"\w+", new_text) if len(w) > 2)
    b = set(w.lower() for w in re.findall(r"\w+", old_text) if len(w) > 2)
    if not a or not b:
        return False
    inter = len(a & b)
    union = len(a | b)
    j = inter / union if union else 0.0
    return j >= DUPLICATE_SIMILARITY_THRESHOLD

# ---------------- WebRTC transformer (process frames) ----------------
class OCRTransformer(VideoTransformerBase):
    def __init__(self):
        self.frame_count = 0
        self.last_ocr_text = ""
        self.running = True

    def transform(self, frame):
        """
        Called for each received video frame. We will run OCR for every Nth frame.
        """
        # Convert frame to numpy array (BGR)
        img = frame.to_ndarray(format="bgr24")
        self.frame_count += 1

        # Only run OCR every N frames
        if self.frame_count % OCR_EVERY_N_FRAMES == 0:
            try:
                # Convert to PIL for Google Vision
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                pil_im = Image.fromarray(rgb)
                text = google_ocr_from_pil(pil_im)

                # quick length check
                if text and len(text) >= MIN_OCR_TEXT_LEN:
                    # Deduplicate vs last saved global text
                    with scan_lock:
                        prev = last_text.get("value", "")
                        if not is_duplicate_text(text, prev):
                            parsed = extract_fields_from_text(text)
                            parsed['Timestamp'] = datetime.utcnow().isoformat()
                            scan_results.append({"text": text, "parsed": parsed})
                            last_text["value"] = text
            except Exception as e:
                # Don't crash transformer on OCR errors; just continue
                print("OCR error in transformer:", str(e))

        # (Optionally) overlay some indicator on frame
        overlay = img.copy()
        h, w = overlay.shape[:2]
        cv2.putText(overlay, "Auto-scan ON", (10, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        return overlay

# ---------------- Streamlit UI ----------------
st.title("📇 Auto Business Card Scanner — Google Vision OCR (Auto-scan mode)")

left, right = st.columns([2, 1])
with left:
    st.header("Live Camera (Auto-scan)")
    st.markdown("Hold the business card in front of your camera. The app will automatically scan and add entries to the table below.")
    # Start the webrtc streamer; transformer runs in background
    rtc_config = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})
from streamlit_webrtc import WebRtcMode

webrtc_ctx = webrtc_streamer(
    key="business-card-webrtc",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=rtc_config,
    video_transformer_factory=OCRTransformer,
    media_stream_constraints={"video": True, "audio": False},
    async_transform=True,
)

with right:
    st.header("Controls")
    st.write(f"- OCR every **{OCR_EVERY_N_FRAMES}** frames")
    st.write(f"- Duplicate similarity threshold: **{DUPLICATE_SIMILARITY_THRESHOLD}**")
    auto_save = st.checkbox("Auto-save detected entries to scans.csv", value=True)
    show_raw = st.checkbox("Show raw OCR (latest)", value=False)
    clear_button = st.button("Clear saved session entries (not file)")

# Show live parsed results (read from scan_results)
st.markdown("---")
st.subheader("Detected entries (live)")
with scan_lock:
    # Move new parsed entries from global store into session_state so Streamlit displays them
    if "detected" not in st.session_state:
        st.session_state.detected = []
    # Move all scan_results parsed into session list
    while scan_results:
        item = scan_results.pop(0)
        parsed = item["parsed"]
        st.session_state.detected.insert(0, parsed)  # newest first
        if auto_save:
            save_entry(parsed)

# If clear pressed, clear session_state.detected
if clear_button:
    st.session_state.detected = []
    st.success("Cleared current session detections (scans.csv file not deleted).")

# Display table
detected_df = pd.DataFrame(st.session_state.get("detected", []))
if not detected_df.empty:
    # Show latest first
    st.dataframe(detected_df[['Name','Company','Role','Phone','Email','Website','Timestamp']].fillna(''), use_container_width=True)
    csv = detected_df.to_csv(index=False).encode('utf-8')
    st.download_button("Download CSV of current session", csv, file_name="scans_session.csv", mime="text/csv")
else:
    st.info("No cards detected yet. Hold a card in front of the camera. Scans will appear automatically.")

# Show (optional) latest raw OCR text
if show_raw and "detected" in st.session_state and st.session_state.detected:
    st.markdown("### Latest parsed OCR text (raw)")
    st.text_area("raw_ocr", value=last_text.get("value", ""), height=180)

st.markdown("---")
st.caption("Works on Streamlit Cloud — requires Google Vision service account JSON in Streamlit Secrets. "
           "If you want faster scanning or better parsing, I can add auto-cropping and perspective correction next.")

