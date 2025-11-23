import streamlit as st
import requests
import base64
import pandas as pd
import re
from datetime import datetime

st.set_page_config(page_title="Business Card Scanner — DeepSeek Vision V4", layout="wide")

# ---------------- DEEPSEEK API ----------------
API_KEY = st.secrets["deepseek"]["api_key"]
API_URL = "https://api.deepseek.com/v1/chat/completions"

def deepseek_vision_ocr(image_bytes):
    # Convert to base64
    encoded = base64.b64encode(image_bytes).decode("utf-8")

    # Correct format for DeepSeek Vision
    payload = {
        "model": "deepseek-chat",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Extract ONLY the raw text from this business card. No formatting. No explanation."
                    },
                    {
                        "type": "image_url",
                        "image_url": f"data:image/png;base64,{encoded}"
                    }
                ]
            }
        ],
        "temperature": 0
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }

    try:
        r = requests.post(API_URL, headers=headers, json=payload, timeout=60)
        data = r.json()
    except Exception as e:
        return f"OCR Error: Request failed → {e}"

    # If DeepSeek returns an error
    if "error" in data:
        return f"OCR Error: {data['error']}"

    try:
        return data["choices"][0]["message"]["content"]
    except:
        return "OCR Error: Failed to read response"


# ---------------- REGEX PATTERNS ----------------
PHONE_REGEX = re.compile(r"(\+?\d[\d\-\s\(\)]{6,}\d)")
EMAIL_REGEX = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
WEBSITE_REGEX = re.compile(r"(https?://\S+|www\.\S+|\b\w+\.(com|in|net|io|co|org))")


# ---------------- EXTRACTION V4 (SAFE) ----------------
def extract_v4(text):
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

    # SAFE extraction — no more index errors
    phones = PHONE_REGEX.findall(block)
    phones = [re.sub(r"[^0-9+]", "", p) for p in phones]

    primary = phones[0] if len(phones) > 0 else ""
    secondary = phones[1] if len(phones) > 1 else ""

    emails = EMAIL_REGEX.findall(block)
    email = emails[0] if len(emails) > 0 else ""

    webs = WEBSITE_REGEX.findall(block)
    website = webs[0][0] if len(webs) > 0 else ""

    # AI name/company extraction
    prompt = f"""
    Extract the following from the business card text below:

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
        "messages": [{"role": "user", "content": prompt}],
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


# ---------------- CSV STORAGE ----------------
def load_csv():
    try:
        return pd.read_csv("scans.csv")
    except:
        return pd.DataFrame(columns=[
            "Name","Company","Role","PhonePrimary","PhoneSecondary",
            "Email","Website","RawText","Timestamp"
        ])

def save_row(row):
    df = load_csv()
    df.loc[len(df)] = row
    df.to_csv("scans.csv", index=False)


# ---------------- UI ----------------
st.title("📇 Business Card Scanner — DeepSeek Vision V4 (Stable)")

col1, col2 = st.columns([2, 1])
with col1:
    cam = st.camera_input("📷 Take a Photo")
    upload = st.file_uploader("📁 Upload Card", type=["jpg","jpeg","png"])

with col2:
    auto = st.checkbox("Auto-save", value=True)
    clear = st.button("Clear All")

image_bytes = None
if cam:
    image_bytes = cam.getvalue()
elif upload:
    image_bytes = upload.read()

if image_bytes:
    st.image(image_bytes, caption="Card", use_column_width=True)

    with st.spinner("🧠 DeepSeek Vision OCR running..."):
        raw_text = deepseek_vision_ocr(image_bytes)

    st.text_area("📄 OCR Output", raw_text, height=200)

    parsed = extract_v4(raw_text)
    st.json(parsed)

    if auto:
        save_row(parsed)
        st.success("Saved!")

if clear:
    df = load_csv()
    df.iloc[0:0].to_csv("scans.csv", index=False)
    st.success("Cleared!")

df = load_csv()
st.subheader("📚 Saved Data")
st.dataframe(df, use_container_width=True)

csv = df.to_csv(index=False).encode()
st.download_button("📥 Download CSV", csv, "cards.csv", "text/csv")
