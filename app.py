import streamlit as st
import requests
import base64
import pandas as pd
import re
import json
import io
from datetime import datetime

st.set_page_config(page_title="Business Card Scanner", layout="wide")

# ---------------- CONFIG ----------------
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
        
        # Updated API format for DeepSeek Vision
        payload = {
            "model": "deepseek-vl",
            "messages": [
                {
                    "role": "user", 
                    "content": [
                        {
                            "type": "text", 
                            "text": "Extract all text from this business card image. Return ONLY the raw text with line breaks, no explanations."
                        },
                        {
                            "type": "image_url",
                            "image_url": f"data:image/jpeg;base64,{encoded}"
                        }
                    ]
                }
            ],
            "temperature": 0.1,
            "max_tokens": 2000,
            "stream": False
        }

        headers = {
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json"
        }

        response = requests.post(API_URL, headers=headers, json=payload, timeout=60)
        
        if response.status_code != 200:
            return f"API Error {response.status_code}: {response.text}"
        
        data = response.json()
        
        # Extract the response content
        if "choices" in data and len(data["choices"]) > 0:
            return data["choices"][0]["message"]["content"]
        else:
            return f"Unexpected API response: {json.dumps(data, indent=2)}"
        
    except Exception as e:
        return f"OCR Error: {str(e)}"

# ---------------- FIELD EXTRACTION ----------------
PHONE_REGEX = re.compile(r"(\+?\d[\d\-\s\(\)]{7,}\d)")
EMAIL_REGEX = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
WEBSITE_REGEX = re.compile(r"(https?://[^\s]+|www\.[^\s]+|[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})")

def extract_fields_simple(text):
    """Simple regex-based field extraction"""
    if text.startswith("OCR Error") or text.startswith("API Error"):
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
        # Clean phone number
        phone_clean = re.sub(r'[^\d+]', '', phones[0])
        data["Phone"] = phone_clean
    
    emails = EMAIL_REGEX.findall(text)
    if emails:
        data["Email"] = emails[0]
    
    websites = WEBSITE_REGEX.findall(text)
    if websites:
        # Take the first website that looks legitimate
        for web in websites:
            if len(web) > 5 and '.' in web:  # Basic website validation
                data["Website"] = web
                break
    
    # Simple heuristics for company and role
    if len(lines) > 1:
        # Skip name line and look for company/role
        for i in range(1, min(5, len(lines))):
            line = lines[i]
            line_lower = line.lower()
            
            # Role indicators
            role_indicators = ['manager', 'director', 'president', 'ceo', 'cto', 'cfo', 
                              'engineer', 'analyst', 'specialist', 'consultant', 'officer',
                              'head', 'lead', 'developer', 'designer']
            
            # Company indicators  
            company_indicators = ['inc', 'corp', 'llc', 'ltd', 'company', 'co.', 'group', 'technologies']
            
            if any(role in line_lower for role in role_indicators) and not data["Role"]:
                data["Role"] = line
            elif any(company in line_lower for company in company_indicators) and not data["Company"]:
                data["Company"] = line
            elif not data["Company"] and len(line.split()) <= 4 and line.upper() == line:
                # All caps short lines are often company names
                data["Company"] = line
    
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
st.title("📇 Business Card Scanner")

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
if image_bytes and st.button("🔍 Scan Business Card", type="primary"):
    with st.spinner("Scanning with DeepSeek Vision..."):
        raw_text = deepseek_ocr(image_bytes)
        
        st.subheader("📄 OCR Results")
        
        if raw_text and not raw_text.startswith("OCR Error") and not raw_text.startswith("API Error"):
            st.success("✅ OCR Successful!")
            
            # Show raw text
            with st.expander("📋 Raw Extracted Text"):
                st.text_area("Full text", raw_text, height=200, key="raw_text")
            
            # Extract fields
            parsed_data = extract_fields_simple(raw_text)
            
            # Display extracted fields
            st.subheader("👤 Extracted Information")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if parsed_data["Name"]:
                    st.text_input("Name", parsed_data["Name"], key="name_display")
                if parsed_data["Company"]:
                    st.text_input("Company", parsed_data["Company"], key="company_display")
                if parsed_data["Role"]:
                    st.text_input("Role", parsed_data["Role"], key="role_display")
            
            with col2:
                if parsed_data["Phone"]:
                    st.text_input("Phone", parsed_data["Phone"], key="phone_display")
                if parsed_data["Email"]:
                    st.text_input("Email", parsed_data["Email"], key="email_display")
                if parsed_data["Website"]:
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
            st.info("Please check your API key and try again.")

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

# Debug info
with st.expander("🔧 Debug Information"):
    st.write("**API Status:**", "Configured" if API_KEY else "Not Configured")
    if st.button("Test API Connection"):
        try:
            headers = {"Authorization": f"Bearer {API_KEY}"}
            response = requests.get("https://api.deepseek.com/user/balance", headers=headers)
            if response.status_code == 200:
                st.success("✅ API Connection Successful!")
                st.json(response.json())
            else:
                st.error(f"❌ API Connection Failed: {response.status_code}")
        except Exception as e:
            st.error(f"❌ API Test Error: {e}")
