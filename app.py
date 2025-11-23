import streamlit as st
import requests
import base64
import pandas as pd
import re
import json
import io
from datetime import datetime

st.set_page_config(page_title="Business Card Scanner — DeepSeek Vision", layout="wide")

# ---------------- CONFIG ----------------
# Check if API key exists, if not, show instructions
if "deepseek" not in st.secrets or "api_key" not in st.secrets["deepseek"]:
    st.error("🔑 DeepSeek API Key not found in secrets!")
    st.info("""
    **To set up your API key:**
    1. Get a DeepSeek API key from https://platform.deepseek.com/
    2. In Streamlit Cloud, go to Settings → Secrets
    3. Add this:
    ```
    [deepseek]
    api_key = "your_actual_api_key_here"
    ```
    """)
    st.stop()

API_KEY = st.secrets["deepseek"]["api_key"]
API_URL = "https://api.deepseek.com/v1/chat/completions"

# ---------------- DEEPSEEK OCR ----------------
def deepseek_ocr(image_bytes):
    """Use DeepSeek Vision API for OCR"""
    try:
        # Convert image to base64
        encoded = base64.b64encode(image_bytes).decode("utf-8")
        
        prompt = """
You are a business card OCR specialist. Extract ALL text from this business card image exactly as it appears, preserving line breaks and formatting. Return ONLY the raw text, no explanations.
"""

        payload = {
            "model": "deepseek-vl",
            "messages": [
                {
                    "role": "user", 
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded}"}}
                    ]
                }
            ],
            "temperature": 0.1,
            "max_tokens": 1000
        }

        headers = {
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json"
        }

        response = requests.post(API_URL, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        return data["choices"][0]["message"]["content"]
        
    except Exception as e:
        return f"OCR Error: {str(e)}"

# ---------------- FIELD EXTRACTION ----------------
PHONE_REGEX = re.compile(r"(\+?\d[\d\-\s\(\)]{7,}\d)")
EMAIL_REGEX = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
WEBSITE_REGEX = re.compile(r"(https?://[^\s]+|www\.[^\s]+|[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})")

def extract_fields_simple(text):
    """Simple regex-based field extraction (no API calls)"""
    if text.startswith("OCR Error"):
        return {
            "Name": "", "Company": "", "Role": "",
            "Phone": "", "Email": "", "Website": "",
            "RawText": text,
            "Timestamp": datetime.now().isoformat()
        }

    lines = [line.strip() for line in text.split("\n") if line.strip()]
    
    # Initialize data
    data = {
        "Name": lines[0] if lines else "",
        "Company": "",
        "Role": "",
        "Phone": "",
        "Email": "",
        "Website": "",
        "RawText": text,
        "Timestamp": datetime.now().isoformat()
    }
    
    # Extract contact info using regex
    phones = PHONE_REGEX.findall(text)
    if phones:
        data["Phone"] = phones[0]
    
    emails = EMAIL_REGEX.findall(text)
    if emails:
        data["Email"] = emails[0]
    
    websites = WEBSITE_REGEX.findall(text)
    if websites:
        data["Website"] = websites[0]
    
    # Simple heuristics for company and role
    if len(lines) > 1:
        second_line = lines[1]
        # Check if second line might be a role
        role_indicators = ['manager', 'director', 'president', 'ceo', 'cto', 'cfo', 'engineer', 'analyst']
        if any(indicator in second_line.lower() for indicator in role_indicators):
            data["Role"] = second_line
            if len(lines) > 2:
                data["Company"] = lines[2]
        else:
            data["Company"] = second_line
            if len(lines) > 2:
                data["Role"] = lines[2]
    
    return data

# ---------------- DATA MANAGEMENT ----------------
def load_scans():
    """Load existing scans from CSV"""
    try:
        return pd.read_csv("scans.csv")
    except:
        return pd.DataFrame(columns=[
            "Name", "Company", "Role", "Phone", "Email", 
            "Website", "RawText", "Timestamp"
        ])

def save_scan(data):
    """Save a new scan to CSV"""
    try:
        df = load_scans()
        new_df = pd.concat([df, pd.DataFrame([data])], ignore_index=True)
        new_df.to_csv("scans.csv", index=False)
        return True
    except Exception as e:
        st.error(f"Save error: {e}")
        return False

# ---------------- STREAMLIT UI ----------------
st.title("📇 Business Card Scanner — DeepSeek Vision")

# Initialize session state
if 'scanned_data' not in st.session_state:
    st.session_state.scanned_data = None

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📸 Capture Card")
    
    # Camera and file upload
    cam_image = st.camera_input("Take photo with camera")
    uploaded_file = st.file_uploader("Or upload image", type=["jpg", "jpeg", "png"])
    
    # Use whichever image is available
    image_bytes = None
    if cam_image:
        image_bytes = cam_image.getvalue()
        st.image(image_bytes, caption="Camera Capture", use_column_width=True)
    elif uploaded_file:
        image_bytes = uploaded_file.read()
        st.image(image_bytes, caption="Uploaded Image", use_column_width=True)

with col2:
    st.subheader("⚙️ Settings")
    auto_save = st.checkbox("Auto-save scans", value=True)
    
    if st.button("🔄 Clear All Data", type="secondary"):
        try:
            empty_df = pd.DataFrame(columns=["Name", "Company", "Role", "Phone", "Email", "Website", "RawText", "Timestamp"])
            empty_df.to_csv("scans.csv", index=False)
            st.session_state.scanned_data = None
            st.success("All data cleared!")
            st.rerun()
        except Exception as e:
            st.error(f"Clear error: {e}")

# Process image when available
if image_bytes:
    with st.spinner("🔍 Scanning with DeepSeek Vision..."):
        raw_text = deepseek_ocr(image_bytes)
        
        if raw_text and not raw_text.startswith("OCR Error"):
            parsed_data = extract_fields_simple(raw_text)
            st.session_state.scanned_data = parsed_data
            
            if auto_save:
                if save_scan(parsed_data):
                    st.success("✅ Scan saved automatically!")
        else:
            st.error(f"❌ {raw_text}")

# Display results if we have scanned data
if st.session_state.scanned_data:
    data = st.session_state.scanned_data
    
    st.subheader("📄 Extracted Information")
    
    # Display in a nice format
    col1, col2 = st.columns(2)
    
    with col1:
        if data["Name"]:
            st.text_input("👤 Name", data["Name"])
        if data["Company"]:
            st.text_input("🏢 Company", data["Company"])
        if data["Role"]:
            st.text_input("💼 Role", data["Role"])
    
    with col2:
        if data["Phone"]:
            st.text_input("📞 Phone", data["Phone"])
        if data["Email"]:
            st.text_input("📧 Email", data["Email"])
        if data["Website"]:
            st.text_input("🌐 Website", data["Website"])
    
    # Raw OCR text
    with st.expander("📋 Raw OCR Text"):
        st.text_area("Full text extracted", data["RawText"], height=200)
    
    # Manual save button if auto-save is off
    if not auto_save and st.button("💾 Save This Scan"):
        if save_scan(data):
            st.success("✅ Scan saved!")

# Display history
st.subheader("📚 Scan History")
df = load_scans()

if not df.empty:
    # Display without raw text for cleaner view
    display_cols = [col for col in df.columns if col != "RawText"]
    st.dataframe(df[display_cols], use_container_width=True)
    
    # Download option
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        "📥 Download CSV",
        data=csv,
        file_name="business_cards.csv",
        mime="text/csv"
    )
else:
    st.info("No scans yet. Capture a business card to get started!")

# API usage info
with st.expander("ℹ️ About This App"):
    st.markdown("""
    **How it works:**
    - Uses DeepSeek Vision API for OCR
    - Extracts contact info using smart parsing
    - Stores data in CSV format
    - Works with camera or file upload
    
    **Tips for best results:**
    - Good lighting on the card
    - Clear, focused image
    - Hold card steady for camera
    """)
