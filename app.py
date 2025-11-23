import streamlit as st
from PIL import Image
import requests
import pandas as pd
import numpy as np
import io
import re
from datetime import datetime

st.set_page_config(page_title="Business Card Scanner (Stable OCR)", layout="wide")

API_KEY = "helloworld"  # free demo key from OCR.Space

# ---------------- OCR API ----------------
def ocr_space(image_bytes):
    url = "https://api.ocr.space/parse/image"
    r = requests.post(
        url,
        files={"file": ("image.png", image_bytes)},
        data={"apikey": API_KEY}
    )
    result = r.json()
    if result.get("ParsedResults"):
        return result["ParsedResults"][0]["ParsedText"]
    return ""

# ---------------- Extraction ----------------
PHONE_REGEX = re.compile(r"(\+?\d[\d\-\s\(\)]{6,}\d)")
EMAIL_REGEX = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
WEBSITE_REGEX = re.compile(r"(https?://\S+|www\.\S+|\b[A-Za-z0-9-]+\.(com|in|net|io|co|org|biz)\b)")

ROLE_KEYWORDS = [
    "ceo","cto","coo","cfo","founder","director","owner","manager",
    "lead","engineer","marketing","sales","product","executive","md"
]

COMP_SUFFIX = [
    "pvt","private","limited","ltd","llp","inc","co","company","tech",
    "solutions","studio","labs","group"
]

def extract_info(text):
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    block = "\n".join(lines)

    # phone
    phones = PHONE_REGEX.findall(block)
    mobile = phones[0] if phones else ""

    # email
    emails = EMAIL_REGEX.findall(block)
    email = emails[0] if emails else ""

    # website
    webs = WEBSITE_REGEX.findall(block)
    website = webs[0] if webs else ""

    # name
    name = ""
    for l in lines:
        if 2 <= len(l.split()) <= 4 and not any(x in l.lower() for x in ROLE_KEYWORDS + COMP_SUFFIX):
            name = l.title()
            break
    if not name and lines:
        name = lines[0].title()

    # role
    role = ""
    for l in lines:
        if any(k in l.lower() for k in ROLE_KEYWORDS):
            role = l.title()
            break

    # company
    company = ""
    for l in lines:
        if any(s in l.lower() for s in COMP_SUFFIX):
            company = l.title()
            break

    return {
        "Name": name,
        "Company": company,
        "Role": role,
        "Phone": mobile,
        "Email": email,
        "Website": website
    }

# ---------------- UI ----------------
st.title("📇 Business Card Scanner — 100% Stable Version (OCR.Space)")

col1, col2 = st.columns([2,1])

with col1:
    cam = st.camera_input("Take photo")
    upload = st.file_uploader("Upload image", type=["png","jpg","jpeg"])

with col2:
    clear = st.button("Clear session")

if "cards" not in st.session_state:
    st.session_state.cards = []

image = None
if cam:
    image = cam.getvalue()
elif upload:
    image = upload.read()

if image:
    st.image(image, caption="Input", use_container_width=True)

    with st.spinner("Reading text..."):
        text = ocr_space(image)

    st.text_area("OCR Text", text, height=200)

    parsed = extract_info(text)
    parsed["Timestamp"] = datetime.utcnow().isoformat()

    st.session_state.cards.append(parsed)
    st.success("Card added!")

if clear:
    st.session_state.cards = []

df = pd.DataFrame(st.session_state.cards)
st.dataframe(df, use_container_width=True)

csv = df.to_csv(index=False).encode("utf-8")
st.download_button("Download CSV", csv, "cards.csv")
