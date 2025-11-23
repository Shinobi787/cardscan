import streamlit as st
import requests
import base64
import pandas as pd
import numpy as np
import re
from datetime import datetime
from PIL import Image
import io

st.set_page_config(page_title="Business Card Scanner — DeepSeek Vision V3", layout="wide")

# ---------------- DEEPSEEK VISION OCR (OFFICIAL) ----------------
API_KEY = st.secrets["deepseek"]["api_key"]
API_URL = "https://api.deepseek.com/v1/chat/completions"

def deepseek_vision_ocr(image_bytes):
    """
    Sends image to DeepSeek VL model with a strict OCR-only prompt.
    """

    # Encode image in base64
    encoded_image = base64.b64encode(image_bytes).decode('utf-8')

    payload = {
        "model": "deepseek-vl",     # <-- Vision model
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "Extract ONLY the raw text from this business card. No explanations. No formatting."},
                    {
                        "type": "input_image",
                        "image_url": f"data:image/png;base64,{encoded_image}"
                    }
                ]
            }
        ],
        "temperature": 0
    }

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    try:
        r = requests.post(API_URL, headers=headers, json=payload, timeout=60)
    except Exception as e:
        return f"OCR Request Failed: {e}"

    try:
        data = r.json()
    except:
        return "OCR Error: Invalid JSON returned"

    if "choices" not in data:
        return f"OCR Error: {data}"

    return data["choices"][0]["message"]["content"]


# ---------------- REGEX PATTERNS ----------------
PHONE_REGEX = re.compile(r"(\+?\d[\d\-\s\(\)]{6,}\d)")
EMAIL_REGEX = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
WEBSITE_REGEX = re.compile(
    r"(https?://\S+|www\.\S+|\b[A-Za-z0-9-]+\.(com|in|net|io|co|org|biz)\b)"
)

ROLE_KEYWORDS = [
    "ceo","cto","coo","cfo","founder","director","owner","manager",
    "lead","engineer","marketing","sales","product","executive","md","gm"
]

COMP_SUFFIX = [
    "pvt","private","limited","ltd","llp","inc","co","company","tech",
    "technologies","solutions","studio","labs","group","enterprise"
]


# ---------------- EXTRACTION V3 (AI + Rules) ----------------
def extract_v3(text):
    """
    Hybrid extraction:
    - Regex and rule-based extraction
    - PLUS AI cleaning & correction via DeepSeek
    """

    # 1) Rule-based parsing (from V2)
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    block = " ".join(lines)

    # PHONE
    phones = PHONE_REGEX.findall(block)
    phones = [re.sub(r"[^0-9+]", "", p) for p in phones]
    primary = secondary = ""

    for p in phones:
        digits = re.sub(r"\D", "", p)
        if len(digits) >= 10:
            if not primary:
                primary = p
            else:
                secondary = p

    # EMAIL
    emails = EMAIL_REGEX.findall(block)
    email = emails[0] if emails else ""

    # WEBSITE
    webs = WEBSITE_REGEX.findall(block)
    website = webs[0][0] if isinstance(webs[0], tuple) else webs[0] if webs else ""

    # 2) AI-based field correction
    correction_prompt = f"""
    The following text was extracted from a business card:

    {text}

    Extract these fields accurately:
    - Full Name
    - Company Name
    - Role / Position
    - Primary Phone
    - Secondary Phone
    - Email
    - Website

    Respond ONLY in valid JSON with keys:
    name, company, role, phone1, phone2, email, website
    """

    payload = {
        "model": "deepseek-chat",
        "messages": [{"role": "user", "content": correction_prompt}],
        "temperature": 0
    }

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    try:
        r = requests.post(API_URL, headers=headers, json=payload)
        j = r.json()
        raw = j["choices"][0]["message"]["content"]
    except:
        # fallback to rule-based only
        return {
            "Name": "",
            "Company": "",
            "Role": "",
            "PhonePrimary": primary,
            "PhoneSecondary": secondary,
            "Email": email,
            "Website": website,
            "RawText": text,
            "Timestamp": datetime.utcnow().isoformat()
        }

    # Parse JSON from DeepSeek
    try:
        ai = eval(raw)  # safe because DeepSeek returns pure JSON
    except:
        ai = {}

    return {
        "Name": ai.get("name", ""),
        "Company": ai.get("company", ""),
        "Role": ai.get("role", ""),
        "PhonePrimary": ai.get("phone1", primary),
        "PhoneSecondary": ai.get("phone2", secondary),
        "Email": ai.get("email", email),
        "Website": ai.get("website", website),
        "RawText": text,
        "Timestamp": datetime.utcnow().isoformat()
    }


# ---------------- CSV STORAGE ----------------
def load_csv():
    try:
        return pd.read_csv("scans.csv")
    except:
        cols = ["Name","Company","Role","PhonePrimary","PhoneSecondary",
                "Email","Website","RawText","Timestamp"]
        return pd.DataFrame(columns=cols)

def save_row(row):
    df = load_csv()
    df.loc[len(df)] = row
    df.to_csv("scans.csv", index=False)


# ---------------- UI ----------------

st.title("📇 Business Card Scanner — DeepSeek Vision OCR (V3 AI Enhanced)")

left, right = st.columns([2,1])

with left:
    cam = st.camera_input("📷 Take Photo of Business Card")
    upload = st.file_uploader("📁 Upload Business Card", type=["png","jpg","jpeg"])

with right:
    auto = st.checkbox("Auto-Save to CSV", value=True)
    clear = st.button("Clear All Records")

image_bytes = None

if cam:
    image_bytes = cam.getvalue()
elif upload:
    image_bytes = upload.read()

if image_bytes:
    st.image(image_bytes, caption="Card Image", use_column_width=True)

    with st.spinner("🧠 DeepSeek Vision Extracting Text..."):
        raw_text = deepseek_ocr(image_bytes)

    st.text_area("📄 OCR Output", raw_text, height=200)

    parsed = extract_v3(raw_text)
    st.json(parsed)

    if auto:
        save_row(parsed)
        st.success("Saved Successfully!")

if clear:
    df = load_csv()
    df.iloc[0:0].to_csv("scans.csv", index=False)
    st.success("All Records Cleared!")

df = load_csv()
st.subheader("📚 Saved Entries")
st.dataframe(df, use_container_width=True)

csv = df.to_csv(index=False).encode()
st.download_button("📥 Download CSV", csv, "cards.csv", "text/csv")
