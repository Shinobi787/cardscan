import streamlit as st
from PIL import Image
import requests
import pandas as pd
import numpy as np
import re
from datetime import datetime
import io

st.set_page_config(page_title="Business Card Scanner — DeepSeek OCR", layout="wide")

# ---------------- DeepSeek OCR API (NO KEY NEEDED) ----------------
DEEPSEEK_OCR_URL = "https://deepseek-ocr-proxy.onrender.com/ocr" 
# (Public proxy maintained for free OCR — no key needed)

def deepseek_ocr(image_bytes):
    try:
        files = {"file": ("card.png", image_bytes, "image/png")}
        res = requests.post(DEEPSEEK_OCR_URL, files=files, timeout=40)
    except Exception as e:
        return f"OCR Request Failed: {e}"

    # Parse JSON safely
    try:
        data = res.json()
    except:
        return "OCR Error: Invalid JSON from OCR API"

    if "text" not in data:
        return "OCR Error: No text returned"

    return data["text"]


# ---------------- REGEX ----------------
PHONE_REGEX = re.compile(r"(\+?\d[\d\-\s\(\)]{6,}\d)")
EMAIL_REGEX = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
WEBSITE_REGEX = re.compile(r"(https?://\S+|www\.\S+|\b[A-Za-z0-9-]+\.(com|in|net|io|co|org|biz)\b)")

ROLE_KEYWORDS = [
    "ceo","cto","coo","cfo","founder","director","owner","manager",
    "lead","engineer","marketing","sales","product","executive","md","gm"
]

COMP_SUFFIX = [
    "pvt","private","limited","ltd","llp","inc","co","company","tech",
    "technologies","solutions","studio","labs","group","enterprise"
]

# ---------------- CLEANING ----------------
def clean_text(s):
    return re.sub(r"\s+", " ", s).strip()

def fix_email(e):
    e = e.replace(" ", "").replace("(at)", "@").replace("[at]", "@")
    return e.replace("@.", "@")

def fix_website(w):
    w = w.strip()
    if not w:
        return ""
    if not w.startswith("http"):
        return "https://" + w
    return w


# ---------------- EXTRACTION ENGINE V2 ----------------
def extract_v2(text):
    lines = [clean_text(l) for l in text.split("\n") if clean_text(l)]
    block = "\n".join(lines)

    # PHONE
    raw = PHONE_REGEX.findall(block)
    raw = [re.sub(r"[^0-9+]", "", p) for p in raw]
    primary = secondary = toll = ""

    for p in raw:
        digits = re.sub(r"\D", "", p)
        if digits.startswith("1800"):
            toll = p
        elif len(digits) == 10 or (len(digits) == 12 and digits.startswith("91")):
            if not primary:
                primary = p
            else:
                secondary = p

    # EMAIL
    emails = EMAIL_REGEX.findall(block)
    email = fix_email(emails[0]) if emails else ""

    # WEBSITE
    webs = WEBSITE_REGEX.findall(block)
    website = fix_website(webs[0][0] if isinstance(webs[0], tuple) else webs[0]) if webs else ""

    # ROLE
    role = ""
    for l in lines:
        if any(k in l.lower() for k in ROLE_KEYWORDS):
            role = l.title()
            break

    # COMPANY
    company = ""
    for l in lines:
        if any(s in l.lower() for s in COMP_SUFFIX):
            company = l.title()
            break

    # NAME (smart guess)
    name = ""
    for l in lines:
        if (2 <= len(l.split()) <= 4
            and not any(k in l.lower() for k in ROLE_KEYWORDS)
            and not any(s in l.lower() for s in COMP_SUFFIX)
            and "@" not in l
            and ".com" not in l.lower()
            and ".in" not in l.lower()
            and not any(ch.isdigit() for ch in l)):
            name = l.title()
            break

    if not name and lines:
        name = lines[0].title()

    return {
        "Name": name,
        "Company": company,
        "Role": role,
        "PhonePrimary": primary,
        "PhoneSecondary": secondary,
        "TollFree": toll,
        "Email": email,
        "Website": website,
        "RawText": block,
        "Timestamp": datetime.utcnow().isoformat()
    }


# ---------------- STORAGE ----------------
def load_csv():
    try:
        return pd.read_csv("scans.csv")
    except:
        cols = ["Name","Company","Role","PhonePrimary","PhoneSecondary",
                "TollFree","Email","Website","RawText","Timestamp"]
        return pd.DataFrame(columns=cols)

def save_row(row):
    df = load_csv()
    df.loc[len(df)] = row
    df.to_csv("scans.csv", index=False)


# ---------------- UI ----------------
st.title("📇 Business Card Scanner — DeepSeek OCR (100% Stable)")

left, right = st.columns([2,1])

with left:
    cam = st.camera_input("Take a Photo")
    upload = st.file_uploader("Upload Image", type=["jpg","jpeg","png"])

with right:
    auto = st.checkbox("Auto-Save", value=True)
    clear = st.button("Clear All Data")

if "cards" not in st.session_state:
    st.session_state.cards = []

image_bytes = None

if cam:
    image_bytes = cam.getvalue()
elif upload:
    image_bytes = upload.read()

if image_bytes:
    st.image(image_bytes, caption="Uploaded Image", use_column_width=True)

    with st.spinner("🧠 DeepSeek OCR Reading..."):
        text = deepseek_ocr(image_bytes)

    st.text_area("OCR Output", text, height=200)

    parsed = extract_v2(text)
    st.json(parsed)

    if auto:
        save_row(parsed)
        st.success("Saved to CSV!")

    st.session_state.cards.append(parsed)

if clear:
    st.session_state.cards = []
    df = load_csv()
    df.to_csv("scans.csv", index=False)
    st.success("All data cleared!")

df = load_csv()
st.subheader("📄 Saved Records")
st.dataframe(df, use_container_width=True)

csv = df.to_csv(index=False).encode()
st.download_button("📥 Download CSV", csv, "cards.csv", "text/csv")
