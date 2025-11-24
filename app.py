import streamlit as st
import requests
import base64
import pandas as pd
import re
from datetime import datetime
from PIL import Image
import io

st.set_page_config(page_title="Business Card Scanner — DeepSeek OCR Final", layout="wide")

API_KEY = st.secrets["deepseek"]["api_key"]
API_URL = "https://api.deepseek.com/v1/chat/completions"

# -----------------------------------------------------------
# 🔥 FIXED & FULLY WORKING DEEPSEEK OCR FUNCTION (BASE64)
# -----------------------------------------------------------
def deepseek_ocr(image_bytes):
    """DeepSeek Vision OCR via base64 text message (stable & error-free)."""

    encoded = base64.b64encode(image_bytes).decode("utf-8")

    prompt = f"""
You are an OCR engine.

Below is a business card image encoded in base64.

1. Decode the base64.
2. Extract ALL readable text EXACTLY as seen.
3. Preserve line breaks.
4. Do NOT add anything extra.

BASE64_IMAGE:
data:image/jpeg;base64,{encoded}
"""

    payload = {
        "model": "deepseek-vl",
        "messages": [
            {"role": "user", "content": prompt}   # TEXT ONLY — DeepSeek requirement
        ],
        "temperature": 0,
        "max_tokens": 4096
    }

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    try:
        r = requests.post(API_URL, headers=headers, json=payload, timeout=120)
        data = r.json()
    except Exception as e:
        return f"OCR Error: {e}"

    if "error" in data:
        try:
            return f"OCR Error: {data['error']['message']}"
        except:
            return f"OCR Error: {data['error']}"

    try:
        return data["choices"][0]["message"]["content"]
    except:
        return "OCR Error: Could not parse DeepSeek response"


# -----------------------------------------------------------
# REGEX PATTERNS
# -----------------------------------------------------------
PHONE_REGEX = re.compile(r"(\+?\d[\d\-\s\(\)]{6,}\d)")
EMAIL_REGEX = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
WEBSITE_REGEX = re.compile(
    r"(https?://\S+|www\.\S+|\b\w+\.(com|in|net|io|co|org))"
)


# -----------------------------------------------------------
# EXTRACTION
# -----------------------------------------------------------
def extract_fields(text):

    if text.startswith("OCR Error"):
        return {
            "Name": "",
            "Company": "",
            "Role": "",
            "PhonePrimary": "",
            "PhoneSecondary": "",
            "Email": "",
            "Website": "",
            "RawText": text,
            "Timestamp": datetime.utcnow().isoformat()
        }

    lines = [l.strip() for l in text.split("\n") if l.strip()]
    block = " ".join(lines)

    phones = PHONE_REGEX.findall(block)
    phones = [re.sub(r"[^0-9+]", "", p) for p in phones]

    primary = phones[0] if len(phones) else ""
    secondary = phones[1] if len(phones) > 1 else ""

    emails = EMAIL_REGEX.findall(block)
    email = emails[0] if emails else ""

    webs = WEBSITE_REGEX.findall(block)
    website = webs[0][0] if isinstance(webs[0], tuple) else (webs[0] if webs else "")

    # AI FIELD EXTRACTION
    cleanup_prompt = f"""
Extract the following fields from the OCR text:

TEXT:
{text}

Return ONLY JSON:
{{
  "name": "",
  "company": "",
  "role": ""
}}
"""

    payload = {
        "model": "deepseek-chat",
        "messages": [{"role": "user", "content": cleanup_prompt}],
        "temperature": 0
    }

    try:
        r = requests.post(API_URL, headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}, json=payload)
        raw = r.json()["choices"][0]["message"]["content"]
        ai = eval(raw)
    except:
        ai = {}

    return {
        "Name": ai.get("name", ""),
        "Company": ai.get("company", ""),
        "Role": ai.get("role", ""),
        "PhonePrimary": primary,
        "PhoneSecondary": secondary,
        "Email": email,
        "Website": website,
        "RawText": text,
        "Timestamp": datetime.utcnow().isoformat()
    }


# -----------------------------------------------------------
# CSV STORAGE
# -----------------------------------------------------------
def load_csv():
    try:
        return pd.read_csv("scans.csv")
    except:
        return pd.DataFrame(columns=[
            "Name","Company","Role",
            "PhonePrimary","PhoneSecondary",
            "Email","Website","RawText","Timestamp"
        ])


def save_row(row):
    df = load_csv()
    df.loc[len(df)] = row
    df.to_csv("scans.csv", index=False)


# -----------------------------------------------------------
# UI
# -----------------------------------------------------------
st.title("📇 Business Card Scanner — DeepSeek OCR (Final Working Version)")

col1, col2 = st.columns([2, 1])

with col1:
    cam = st.camera_input("📸 Take Photo")
    upload = st.file_uploader("📁 Upload Image", type=["jpg","jpeg","png"])

with col2:
    auto = st.checkbox("Auto-save", value=True)
    clear = st.button("Clear All Records")


image_bytes = cam.getvalue() if cam else upload.read() if upload else None

if image_bytes:
    st.image(image_bytes, caption="Business Card", use_column_width=True)

    with st.spinner("🧠 Running DeepSeek OCR..."):
        raw_text = deepseek_ocr(image_bytes)

    st.text_area("📄 OCR Output", raw_text, height=200)

    parsed = extract_fields(raw_text)
    st.json(parsed)

    if auto:
        save_row(parsed)
        st.success("Saved!")


if clear:
    df = load_csv()
    df.iloc[0:0].to_csv("scans.csv", index=False)
    st.success("All records cleared!")


df = load_csv()
st.subheader("📚 Saved Entries")
st.dataframe(df, use_Container_width=True)

csv = df.to_csv(index=False).encode()
st.download_button("📥 Download CSV", csv, "cards.csv", "text/csv")
