# app.py
import streamlit as st
from PIL import Image
import io
import re
import base64
import pandas as pd
import numpy as np
import requests
from datetime import datetime

st.set_page_config(page_title="Business Card Scanner — RapidOCR", layout="wide")

# ---------------- Settings ----------------
# Fallback OCR.space demo key (used only if rapidocr is not available)
OCR_SPACE_KEY = "helloworld"

# A sample uploaded file path that you provided earlier (used only for debug/demo).
# The platform will convert local path to a URL when needed (developer note).
SAMPLE_LOCAL_PATH = "/mnt/data/52f7bca3-4259-45d0-8ca1-ff467481fb56.png"

# ---------------- Try to load RapidOCR ----------------
USE_RAPIDOCR = False
reader = None
try:
    # rapidocr has a simple API: from rapidocr import RapidOCR or rapidocr import Reader
    # Different versions differ; we'll attempt the common import patterns:
    try:
        # new-style
        from rapidocr import RapidOCR
        reader = RapidOCR()
        USE_RAPIDOCR = True
    except Exception:
        from rapidocr import Reader  # fallback
        reader = Reader()
        USE_RAPIDOCR = True
except Exception:
    USE_RAPIDOCR = False

# ---------------- OCR helpers ----------------
def rapidocr_extract(image_bytes):
    """
    Use rapidocr reader to get text. Returns plain text.
    """
    try:
        # reader might accept PIL image or bytes depending on version
        # try both
        pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        try:
            # some RapidOCR versions: reader.readtext(pil) -> list of (box, text, score)
            res = reader.readtext(pil)
        except Exception:
            # other variants return strings
            res = reader.recognize(pil)
        # Normalize result into text
        if isinstance(res, str):
            return res
        texts = []
        for item in res:
            # res may be list of tuples or dicts
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                # many versions: (bbox, text, score)
                texts.append(item[1])
            elif isinstance(item, dict) and "text" in item:
                texts.append(item["text"])
            else:
                # fallback: stringify
                texts.append(str(item))
        return "\n".join([t.strip() for t in texts if t and t.strip()])
    except Exception as e:
        return f"OCR Error (rapidocr): {e}"

def ocr_space_api(image_bytes):
    """
    Fallback OCR using OCR.Space public demo key.
    """
    url = "https://api.ocr.space/parse/image"
    try:
        r = requests.post(
            url,
            files={"file": ("card.png", image_bytes)},
            data={"apikey": OCR_SPACE_KEY},
            timeout=60
        )
    except Exception as e:
        return f"OCR Error (ocr.space): {e}"

    try:
        j = r.json()
    except Exception:
        return "OCR Error (ocr.space): Invalid JSON response"

    if "ParsedResults" in j and j["ParsedResults"]:
        return j["ParsedResults"][0].get("ParsedText", "")
    # return error message if present
    if "ErrorMessage" in j and j["ErrorMessage"]:
        return f"OCR Error (ocr.space): {j['ErrorMessage']}"
    return "OCR Error (ocr.space): No parsed results"

def ocr_dispatch(image_bytes):
    """
    Choose rapidocr if available, otherwise fallback.
    """
    if USE_RAPIDOCR:
        txt = rapidocr_extract(image_bytes)
        # If rapidocr fails with an OCR Error string, fallback
        if isinstance(txt, str) and txt.startswith("OCR Error"):
            return ocr_space_api(image_bytes)
        return txt
    else:
        return ocr_space_api(image_bytes)

# ---------------- Extraction rules (V2 improved) ----------------
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

def clean_text(s):
    return re.sub(r'\s+', ' ', s).strip()

def extract_fields(text):
    if not text or text.startswith("OCR Error"):
        return {
            "Name": "",
            "Company": "",
            "Role": "",
            "Phone": "",
            "SecondPhone": "",
            "Email": "",
            "Website": "",
            "RawText": text,
            "Timestamp": datetime.utcnow().isoformat()
        }

    lines = [clean_text(l) for l in text.split("\n") if clean_text(l)]
    block = "\n".join(lines)

    # phones
    all_phones = PHONE_REGEX.findall(block)
    all_phones = [re.sub(r'[^0-9+]', '', p) for p in all_phones]
    primary = ""
    secondary = ""
    toll = ""
    for p in all_phones:
        digits = re.sub(r"\D","", p)
        if digits.startswith("1800"):
            toll = p
        elif len(digits) >= 10:
            if not primary:
                primary = p
            elif not secondary:
                secondary = p

    # email
    emails = EMAIL_REGEX.findall(block)
    email = emails[0] if emails else ""

    # website
    webs = WEBSITE_REGEX.findall(block)
    website = ""
    if webs:
        first = webs[0]
        if isinstance(first, tuple):
            website = first[0]
        else:
            website = first

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

    # name (smart)
    name = ""
    for l in lines:
        if (2 <= len(l.split()) <= 4 and
            not any(k in l.lower() for k in ROLE_KEYWORDS) and
            not any(s in l.lower() for s in COMP_SUFFIX) and
            "@" not in l and ".com" not in l.lower() and ".in" not in l.lower() and
            not any(ch.isdigit() for ch in l)):
            name = l.title()
            break
    if not name and lines:
        name = lines[0].title()

    return {
        "Name": name,
        "Company": company,
        "Role": role,
        "Phone": primary,
        "SecondPhone": secondary,
        "TollFree": toll,
        "Email": email,
        "Website": website,
        "RawText": block,
        "Timestamp": datetime.utcnow().isoformat()
    }

# ---------------- CSV storage helpers ----------------
def load_csv():
    try:
        return pd.read_csv("scans.csv")
    except Exception:
        return pd.DataFrame(columns=["Name","Company","Role","Phone","SecondPhone","TollFree","Email","Website","RawText","Timestamp"])

def save_row(row):
    df = load_csv()
    df.loc[len(df)] = row
    df.to_csv("scans.csv", index=False)

# ---------------- UI ----------------
st.title("📇 Business Card Scanner — RapidOCR (free)")

col1, col2 = st.columns([2,1])

with col1:
    cam = st.camera_input("📸 Take photo")
    upload = st.file_uploader("Upload image", type=["png","jpg","jpeg"])

with col2:
    auto = st.checkbox("Auto-save to CSV", value=True)
    show_raw = st.checkbox("Show OCR raw text", value=False)
    use_sample = st.button("Use sample image (debug)")

if use_sample:
    # try to load the sample local file you uploaded earlier
    try:
        with open(SAMPLE_LOCAL_PATH, "rb") as f:
            image_bytes = f.read()
    except Exception as e:
        st.error(f"Could not load sample file: {e}")
        image_bytes = None
else:
    image_bytes = None
    if cam:
        image_bytes = cam.getvalue()
    elif upload:
        image_bytes = upload.read()

if image_bytes:
    st.image(image_bytes, caption="Input", use_column_width=True)

    with st.spinner("Running OCR…"):
        ocr_text = ocr_dispatch(image_bytes)

    if show_raw:
        st.text_area("OCR Output", ocr_text, height=250)

    parsed = extract_fields(ocr_text)
    st.subheader("Extracted fields")
    st.json(parsed)

    if auto:
        save_row(parsed)
        st.success("Saved to scans.csv")

else:
    st.info("Take a photo or upload an image of a business card. You can click 'Use sample image (debug)' to try the demo image.")

df = load_csv()
st.subheader("Saved entries")
st.dataframe(df, use_container_width=True)
csv = df.to_csv(index=False).encode("utf-8")
st.download_button("Download CSV", csv, "cards.csv", "text/csv")

# End of app.py
