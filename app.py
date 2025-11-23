import streamlit as st
from PIL import Image, ImageFilter, ImageEnhance, ImageOps
import easyocr
import numpy as np
import pandas as pd
import re
from datetime import datetime
import os

# ---------------- CONFIG ----------------
st.set_page_config(page_title="Business Card Scanner (Stable)", layout="wide")
DATA_FILE = "scans.csv"
SAMPLE_PATH = "/mnt/data/07c1090d-a232-4e82-bb1f-16abd2b9ea93.png"  # your sample image path

# ---------------- OCR ----------------
@st.cache_resource
def load_reader():
    return easyocr.Reader(['en'], gpu=False)

reader = load_reader()

# ---------------- REGEX ----------------
PHONE_REGEX = re.compile(r"(\+?\d[\d\-\s\(\)]{6,}\d)")
EMAIL_REGEX = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
WEBSITE_REGEX = re.compile(r"(https?://\S+|www\.\S+|\b[A-Za-z0-9-]+\.(com|in|net|io|co|org|biz)\b)")

ROLE_KEYWORDS = [
    "ceo","cto","coo","cfo","founder","director","owner","manager","lead",
    "designer","developer","engineer","analyst","consultant","vp","vice",
    "president","marketing","sales","operations","product","executive","md","gm"
]

COMPANY_SUFFIX = [
    "pvt","private","limited","ltd","llp","inc","co","company","technologies",
    "solutions","studio","labs","enterprise","industries","group","stores"
]

# --------------- PREPROCESSING ----------------
def enhance(img: Image.Image):
    img = img.convert("RGB")
    w, h = img.size
    if max(w,h) < 1100:
        scale = int(1100/max(w,h))
        img = img.resize((w*scale, h*scale), Image.LANCZOS)
    img = ImageOps.autocontrast(img)
    img = img.filter(ImageFilter.MedianFilter(3))
    img = ImageEnhance.Sharpness(img).enhance(1.2)
    img = ImageEnhance.Contrast(img).enhance(1.15)
    return img

# --------------- OCR ----------------
def run_ocr(img):
    arr = np.array(img)
    try:
        lines = reader.readtext(arr, detail=0, paragraph=True)
        return "\n".join([l.strip() for l in lines if l.strip()])
    except:
        return ""

# --------------- EXTRACTION ----------------
def extract(text):
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    block = "\n".join(lines)

    # PHONE
    phones = PHONE_REGEX.findall(block)
    cleaned = [re.sub(r'[^0-9+]', '', p) for p in phones]
    mobile = ""
    toll = ""
    for p in cleaned:
        digits = re.sub(r'\D','',p)
        if digits.startswith("1800"):
            toll = p
        if len(digits)==10 or (len(digits)==12 and digits.startswith("91")):
            mobile = p

    # EMAIL
    emails = EMAIL_REGEX.findall(block)
    email = emails[0] if emails else ""

    # WEBSITE
    webs = WEBSITE_REGEX.findall(block)
    website = webs[0][0] if webs else ""

    # NAME
    name = ""
    for l in lines[:5]:
        lw = l.lower()
        if not any(k in lw for k in ROLE_KEYWORDS) and not any(s in lw for s in COMPANY_SUFFIX):
            if 2 <= len(l.split()) <= 4:
                name = l.title()
                break
    if not name and lines:
        name = lines[0].title()

    # ROLE
    role = ""
    for l in lines:
        if any(k in l.lower() for k in ROLE_KEYWORDS):
            role = l.title()
            break

    # COMPANY
    company = ""
    for l in reversed(lines):
        if any(s in l.lower() for s in COMPANY_SUFFIX):
            company = l.title()
            break
    if not company and len(lines) >= 2:
        company = lines[1].title()

    return {
        "Name": name,
        "Company": company,
        "Role": role,
        "Phone": mobile,
        "TollFree": toll,
        "Email": email,
        "Website": website
    }

# --------------- STORAGE ----------------
def load_saved():
    try:
        return pd.read_csv(DATA_FILE)
    except:
        return pd.DataFrame(columns=["Name","Company","Role","Phone","TollFree","Email","Website","Timestamp"])

def save_entry(e):
    df = load_saved()
    df.loc[len(df)] = e
    df.to_csv(DATA_FILE, index=False)

# --------------- UI ----------------
st.title("📇 Business Card Scanner — Stable Version (EasyOCR Only)")

left, right = st.columns([2,1])

with left:
    cam = st.camera_input("Take photo")
    upload = st.file_uploader("Or upload image", type=["jpg","png","jpeg"])
    if st.button("Test sample image"):
        if os.path.exists(SAMPLE_PATH):
            upload = SAMPLE_PATH

with right:
    auto_save = st.checkbox("Auto save", value=True)
    show_raw = st.checkbox("Show OCR text", value=False)
    clear = st.button("Clear session")

if "seen" not in st.session_state:
    st.session_state.seen=[]
if "cards" not in st.session_state:
    st.session_state.cards=[]

img = None
if cam:
    img = Image.open(cam)
elif upload:
    img = Image.open(upload) if not isinstance(upload,str) else Image.open(upload)

if img:
    st.image(img, caption="Original")

    pre = enhance(img)
    st.image(pre, caption="Processed")

    with st.spinner("Running OCR..."):
        text = run_ocr(pre)

    if show_raw:
        st.text_area("OCR Output", text, height=260)

    norm = re.sub(r'[^a-z0-9]','', text.lower())
    if norm not in st.session_state.seen and norm.strip():
        st.session_state.seen.append(norm)
        parsed = extract(text)
        parsed["Timestamp"] = datetime.utcnow().isoformat()
        st.session_state.cards.append(parsed)
        if auto_save:
            save_entry(parsed)
        st.success("Card added!")
    else:
        st.info("Duplicate ignored.")

if clear:
    st.session_state.cards=[]
    st.session_state.seen=[]
    st.success("Session cleared")

df = pd.DataFrame(st.session_state.cards)
saved = load_saved()

final = pd.concat([df, saved], ignore_index=True)
st.subheader("📄 Scanned Cards")
st.dataframe(final, use_container_width=True)

csv = final.to_csv(index=False).encode("utf-8")
st.download_button("Download CSV", csv, "cards.csv", "text/csv")
