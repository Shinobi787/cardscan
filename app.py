import streamlit as st
from PIL import Image, ImageOps, ImageEnhance, ImageFilter
import easyocr
import numpy as np
import pandas as pd
import cv2
import re
from datetime import datetime

st.set_page_config(page_title="Business Card Scanner", layout="wide")

# Load EasyOCR (cached)
@st.cache_resource
def load_ocr():
    return easyocr.Reader(['en'], gpu=False)

ocr = load_ocr()

# Regex
PHONE_REGEX = re.compile(r"(\+?\d[\d\-\s\(\)]{6,}\d)")
EMAIL_REGEX = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
WEBSITE_REGEX = re.compile(r"(https?://\S+|www\.\S+|\b[A-Za-z0-9-]+\.(com|in|net|io|co|org|biz)\b)")

# Preprocess
def enhance(img):
    img = img.convert("RGB")
    img = ImageOps.autocontrast(img)
    img = ImageEnhance.Sharpness(img).enhance(1.2)
    img = ImageEnhance.Contrast(img).enhance(1.1)
    return img

# OCR
def read_text(pil_img):
    img = np.array(pil_img)
    try:
        result = ocr.readtext(img, detail=0, paragraph=True)
        return "\n".join(result)
    except:
        return ""

# Extraction
def extract(text):
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    block = "\n".join(lines)

    phones = PHONE_REGEX.findall(block)
    phone = phones[0] if phones else ""

    emails = EMAIL_REGEX.findall(block)
    email = emails[0] if emails else ""

    webs = WEBSITE_REGEX.findall(block)
    website = webs[0] if webs else ""

    name = ""
    if lines:
        name = lines[0].title()

    company = ""
    if len(lines) > 1:
        company = lines[1].title()

    return {
        "Name": name,
        "Company": company,
        "Phone": phone,
        "Email": email,
        "Website": website,
        "Raw": text,
        "Timestamp": datetime.utcnow().isoformat()
    }

# Storage
def load_csv():
    try:
        return pd.read_csv("scans.csv")
    except:
        return pd.DataFrame(columns=["Name","Company","Phone","Email","Website","Raw","Timestamp"])

def save_row(d):
    df = load_csv()
    df.loc[len(df)] = d
    df.to_csv("scans.csv", index=False)

# UI
st.title("📇 Business Card Scanner (EasyOCR Stable Version)")

col1, col2 = st.columns(2)

with col1:
    cam = st.camera_input("Take a Photo")
    upload = st.file_uploader("Upload image", type=["png","jpg","jpeg"])

image = None
if cam:
    image = Image.open(cam)
elif upload:
    image = Image.open(upload)

if image:
    st.image(image, caption="Original", use_column_width=True)

    pre = enhance(image)
    st.image(pre, caption="Processed", use_column_width=True)

    text = read_text(pre)
    st.text_area("OCR Output", text, height=200)

    data = extract(text)
    st.write("### Extracted Fields")
    st.json(data)

    if st.button("Save"):
        save_row(data)
        st.success("Saved!")

df = load_csv()
st.write("### Saved Scans")
st.dataframe(df, use_container_width=True)
