import streamlit as st
import requests
import base64
import pandas as pd
import re
import json
import io
from datetime import datetime

st.set_page_config(page_title="Business Card Scanner", layout="wide")

# ---------------- FREE OCR API ----------------
def free_ocr(image_bytes):
    """Use free OCR.space API"""
    try:
        # Convert image to base64
        encoded_image = base64.b64encode(image_bytes).decode('utf-8')
        
        # Use OCR.space free API (no key needed for limited use)
        url = "https://api.ocr.space/parse/image"
        payload = {
            'base64Image': f'data:image/jpeg;base64,{encoded_image}',
            'language': 'eng',
            'isOverlayRequired': False,
            'OCREngine': 2
        }
        
        response = requests.post(url, data=payload, timeout=30)
        data = response.json()
        
        if response.status_code == 200 and data['IsErroredOnProcessing'] == False:
            return data['ParsedResults'][0]['ParsedText']
        else:
            return f"OCR Error: {data.get('ErrorMessage', 'Unknown error')}"
            
    except Exception as e:
        return f"OCR Error: {str(e)}"

# Alternative: Use another free OCR service
def alternative_ocr(image_bytes):
    """Alternative free OCR service"""
    try:
        # Convert to base64
        encoded_image = base64.b64encode(image_bytes).decode('utf-8')
        
        # Using a simple OCR API
        api_url = "https://api.ocr.space/parse/image"
        payload = {
            'apikey': 'helloworld',  # Free key
            'base64Image': f'data:image/jpeg;base64,{encoded_image}',
            'language': 'eng',
            'OCREngine': 1
        }
        
        response = requests.post(api_url, data=payload, timeout=30)
        data = response.json()
        
        if data.get('ParsedResults'):
            return data['ParsedResults'][0]['ParsedText']
        else:
            return "OCR Error: No text found"
            
    except Exception as e:
        return f"OCR Error: {str(e)}"

# ---------------- FIELD EXTRACTION ----------------
PHONE_REGEX = re.compile(r"(\+?\d[\d\-\s\(\)]{7,}\d)")
EMAIL_REGEX = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
WEBSITE_REGEX = re.compile(r"(https?://[^\s]+|www\.[^\s]+|[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})")

def extract_fields_simple(text):
    """Simple regex-based field extraction"""
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
            if len(web) > 5 and '.' in web:
                data["Website"] = web
                break
    
    # Simple heuristics for company and role
    if len(lines) > 1:
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
st.markdown("**Free OCR - No API Key Required**")

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
    with st.spinner("Scanning with OCR..."):
        # Try primary OCR first, fallback to alternative
        raw_text = free_ocr(image_bytes)
        
        # If first OCR fails, try alternative
        if raw_text.startswith("OCR Error"):
            raw_text = alternative_ocr(image_bytes)
        
        st.subheader("📄 OCR Results")
        
        if raw_text and not raw_text.startswith("OCR Error"):
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
            st.info("""
            **Troubleshooting tips:**
            - Make sure the image is clear and well-lit
            - Try a different image format (JPG works best)
            - Ensure the text is readable and not blurry
            """)

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

# Manual entry fallback
with st.expander("✍️ Manual Entry (Fallback)"):
    st.write("If OCR doesn't work well, you can manually enter the details:")
    
    with st.form("manual_entry"):
        manual_name = st.text_input("Name")
        manual_company = st.text_input("Company")
        manual_role = st.text_input("Role")
        manual_phone = st.text_input("Phone")
        manual_email = st.text_input("Email")
        manual_website = st.text_input("Website")
        
        if st.form_submit_button("Add Manual Entry"):
            if manual_name:
                manual_data = {
                    "Name": manual_name,
                    "Company": manual_company,
                    "Role": manual_role,
                    "Phone": manual_phone,
                    "Email": manual_email,
                    "Website": manual_website,
                    "RawText": "Manually entered",
                    "Timestamp": datetime.now().isoformat()
                }
                if save_scan(manual_data):
                    st.success("✅ Manual entry saved!")
                    st.rerun()
            else:
                st.error("Please at least enter a name")

st.markdown("---")
st.caption("This app uses free OCR services - no API key required!")
