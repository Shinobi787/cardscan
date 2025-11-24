import streamlit as st
import requests
import base64
import pandas as pd
import re
import json
import io
from datetime import datetime

st.set_page_config(page_title="Business Card Scanner — DeepSeek Vision", layout="wide")

# ---------------- DEEPSEEK API CONFIG ----------------
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

# ---------------- CORRECT DEEPSEEK VISION API CALL ----------------
def deepseek_ocr(image_bytes):
    """DeepSeek Vision OCR via base64 text message using deepseek-chat model."""

    # Encode image to base64
    encoded = base64.b64encode(image_bytes).decode("utf-8")

    # Strict OCR prompt
    prompt = f"""
You are an OCR engine with perfect vision.

Below is a business card image encoded as base64.

TASK:
1. Decode the base64 image.
2. Read ALL text from the image exactly as it appears.
3. Preserve line breaks.
4. Do NOT explain. Do NOT add extra words.
5. Return ONLY the raw text content.

BASE64_IMAGE:
data:image/jpeg;base64,{encoded}
"""

    payload = {
        "model": "deepseek-chat",  # Using available model
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": 0,
        "max_tokens": 2048
    }

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=120)
        data = response.json()
    except Exception as e:
        return f"OCR Error: {e}"

    # Handle DeepSeek API errors
    if "error" in data:
        try:
            return f"OCR Error: {data['error']['message']}"
        except:
            return f"OCR Error: {data['error']}"

    # Extract text from DeepSeek output
    try:
        return data["choices"][0]["message"]["content"]
    except:
        return "OCR Error: Could not parse DeepSeek response"

# ---------------- IMPROVED FIELD EXTRACTION ----------------
PHONE_REGEX = re.compile(r"(\+?\d[\d\-\s\(\)]{7,}\d)")
EMAIL_REGEX = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
WEBSITE_REGEX = re.compile(r"(https?://[^\s]+|www\.[^\s]+|[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})")

def extract_fields_advanced(text):
    """Advanced field extraction with better heuristics"""
    if any(error in text for error in ["OCR Error", "API Error"]):
        return {
            "Name": "", "Company": "", "Role": "",
            "Phone": "", "Email": "", "Website": "",
            "RawText": text,
            "Timestamp": datetime.now().isoformat()
        }

    lines = [line.strip() for line in text.split("\n") if line.strip()]
    
    data = {
        "Name": "",
        "Company": "",
        "Role": "",
        "Phone": "",
        "Email": "",
        "Website": "",
        "RawText": text,
        "Timestamp": datetime.now().isoformat()
    }
    
    # Extract contact info
    phones = PHONE_REGEX.findall(text)
    if phones:
        data["Phone"] = re.sub(r'[^\d+]', '', phones[0])
    
    emails = EMAIL_REGEX.findall(text)
    if emails:
        data["Email"] = emails[0]
    
    websites = WEBSITE_REGEX.findall(text)
    if websites:
        for web in websites[:3]:  # Check first 3 matches
            clean_web = web if isinstance(web, str) else web[0]
            if len(clean_web) > 5 and '.' in clean_web and '@' not in clean_web:
                data["Website"] = clean_web
                break
    
    # Advanced name/company/role extraction
    if lines:
        # Name is usually first non-empty line
        data["Name"] = lines[0]
        
        # Analyze remaining lines
        remaining_lines = lines[1:]
        
        for i, line in enumerate(remaining_lines):
            line_lower = line.lower()
            
            # Role detection
            role_patterns = [
                r'\b(manager|director|president|ceo|cto|cfo|vp|vice president|engineer|analyst|specialist|consultant|officer|head|lead|developer|designer|architect)\b',
                r'\b(senior|junior|principal|staff)\s+\w+',
                r'\b(head|chief|lead)\s+of\s+\w+'
            ]
            
            for pattern in role_patterns:
                if re.search(pattern, line_lower, re.IGNORECASE):
                    data["Role"] = line
                    break
            
            # Company detection
            company_indicators = ['inc', 'corp', 'llc', 'ltd', 'company', 'co.', 'group', 'technologies', 'solutions', 'systems', 'enterprises']
            if any(indicator in line_lower for indicator in company_indicators):
                data["Company"] = line
            elif not data["Company"] and (line.isupper() or any(word.istitle() for word in line.split())):
                # Company names often have proper capitalization or are all caps
                if len(line.split()) <= 4:  # Reasonable company name length
                    data["Company"] = line
            
            # Stop after checking a few lines
            if i >= 4:
                break
    
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
st.title("📇 Business Card Scanner — DeepSeek")
st.markdown("**Powered by DeepSeek AI**")

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
    elif uploaded_file:
        image_bytes = uploaded_file.read()
    
    if image_bytes:
        st.image(image_bytes, caption="Image to Scan", use_column_width=True)

with col2:
    st.subheader("⚙️ Settings")
    auto_save = st.checkbox("Auto-save scans", value=True)
    
    if st.button("🔄 Clear All Data"):
        try:
            empty_df = pd.DataFrame(columns=["Name", "Company", "Role", "Phone", "Email", "Website", "RawText", "Timestamp"])
            empty_df.to_csv("scans.csv", index=False)
            st.success("All data cleared!")
            st.rerun()
        except Exception as e:
            st.error(f"Clear error: {e}")

# Process image when available
if image_bytes and st.button("🔍 Scan with DeepSeek", type="primary"):
    with st.spinner("🔄 Calling DeepSeek API..."):
        raw_text = deepseek_ocr(image_bytes)
        
        st.subheader("📄 OCR Results")
        
        if raw_text and not any(error in raw_text for error in ["OCR Error", "API Error"]):
            st.success("✅ DeepSeek OCR Successful!")
            
            # Show raw text
            with st.expander("📋 Raw Extracted Text"):
                st.text_area("Full text", raw_text, height=200, key="raw_text")
            
            # Extract fields
            parsed_data = extract_fields_advanced(raw_text)
            
            # Display extracted fields
            st.subheader("👤 Extracted Information")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.text_input("Name", parsed_data["Name"], key="name_display")
                st.text_input("Company", parsed_data["Company"], key="company_display")
                st.text_input("Role", parsed_data["Role"], key="role_display")
            
            with col2:
                st.text_input("Phone", parsed_data["Phone"], key="phone_display")
                st.text_input("Email", parsed_data["Email"], key="email_display")
                st.text_input("Website", parsed_data["Website"], key="website_display")
            
            # Save data
            if auto_save:
                if save_scan(parsed_data):
                    st.success("✅ Scan saved automatically!")
            else:
                if st.button("💾 Save This Scan"):
                    if save_scan(parsed_data):
                        st.success("✅ Scan saved!")
        else:
            st.error(f"❌ {raw_text}")
            
            # Debug information
            with st.expander("🔧 Debug Information"):
                st.write("**API Key Status:**", "Configured" if API_KEY else "Missing")
                st.write("**Image Size:**", f"{len(image_bytes)} bytes")
                
                if st.button("Test DeepSeek API Connection"):
                    try:
                        headers = {"Authorization": f"Bearer {API_KEY}"}
                        # Test with a simple text request
                        test_payload = {
                            "model": "deepseek-chat",
                            "messages": [{"role": "user", "content": "Say 'API is working'"}],
                            "max_tokens": 10
                        }
                        response = requests.post(API_URL, headers=headers, json=test_payload, timeout=30)
                        if response.status_code == 200:
                            st.success("✅ DeepSeek API Connection Successful!")
                        else:
                            st.error(f"❌ API Test Failed: {response.status_code} - {response.text}")
                    except Exception as e:
                        st.error(f"❌ API Test Error: {e}")

# Display history
st.subheader("📚 Scan History")
df = load_scans()

if not df.empty:
    # Display without raw text for cleaner view
    display_cols = [col for col in df.columns if col != "RawText"]
    st.dataframe(df[display_cols].sort_values("Timestamp", ascending=False), 
                use_container_width=True,
                height=400)
    
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

# API status
with st.expander("ℹ️ DeepSeek Status"):
    st.markdown("""
    **DeepSeek Features:**
    - OCR via base64 image embedding
    - Handles various fonts and layouts
    - Extracts text from business cards
    - Supports multiple languages
    
    **Current Status:** ✅ Configured
    **Model:** deepseek-chat
    **Endpoint:** https://api.deepseek.com/v1/chat/completions
    """)
