import streamlit as st
import pandas as pd
import re
import easyocr
import os
from datetime import datetime
from PIL import Image
import io
import numpy as np

# Page config
st.set_page_config(
    page_title="Business Card Scanner",
    page_icon="📇",
    layout="wide"
)

# Configuration
DATA_FILE = "scans.csv"
MIN_TEXT_LENGTH = 10

# Regex patterns
PHONE_REGEX = re.compile(r'(\+?\d[\d\s\-\(\)]{7,}\d)')
EMAIL_REGEX = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
WEBSITE_REGEX = re.compile(r'(https?://[^\s]+|www\.[^\s]+|\b[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b)')

@st.cache_resource
def init_ocr():
    """Initialize EasyOCR reader"""
    try:
        reader = easyocr.Reader(['en'])
        st.success("✅ OCR loaded successfully!")
        return reader
    except Exception as e:
        st.error(f"❌ OCR failed to load: {str(e)}")
        return None

def extract_fields_from_text(text):
    """Extract business card fields from OCR text"""
    if text is None:
        text = ""
        
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    
    data = {
        'Name': '',
        'Company': '',
        'Role': '',
        'Phone': '',
        'Email': '',
        'Website': '',
        'RawText': text,
        'Timestamp': datetime.now().isoformat()
    }
    
    if not lines:
        return data
    
    # Extract emails
    emails = EMAIL_REGEX.findall(text)
    if emails:
        data['Email'] = emails[0]
    
    # Extract phones
    phones = PHONE_REGEX.findall(text)
    if phones:
        data['Phone'] = max(phones, key=len)
    
    # Extract websites
    websites = WEBSITE_REGEX.findall(text)
    if websites:
        for web in websites:
            if not any(email in web for email in emails if emails):
                data['Website'] = web
                break
    
    # Basic field extraction
    if lines:
        data['Name'] = lines[0]
        
        if len(lines) > 1:
            data['Company'] = lines[1]
        
        # Look for role indicators in remaining lines
        role_indicators = ['manager', 'director', 'president', 'ceo', 'cto', 'cfo', 'engineer']
        for line in lines[2:5]:
            if any(role in line.lower() for role in role_indicators):
                data['Role'] = line
                break
    
    return data

def load_scans():
    """Load previous scans from CSV"""
    if os.path.exists(DATA_FILE):
        try:
            df = pd.read_csv(DATA_FILE)
            return df.fillna('').to_dict('records')
        except:
            return []
    return []

def save_scan(data):
    """Save scan to CSV"""
    try:
        df = pd.DataFrame([data])
        
        if os.path.exists(DATA_FILE):
            existing_df = pd.read_csv(DATA_FILE)
            df = pd.concat([existing_df, df], ignore_index=True)
        
        df.to_csv(DATA_FILE, index=False)
        return True
    except Exception as e:
        st.error(f"Error saving: {e}")
        return False

def create_sample_card():
    """Create a sample business card image for testing"""
    img = Image.new('RGB', (400, 250), color='white')
    
    # Simple text drawing without font issues
    from PIL import ImageDraw
    d = ImageDraw.Draw(img)
    
    d.text((20, 20), "John Smith", fill='black')
    d.text((20, 50), "Marketing Director", fill='black')
    d.text((20, 80), "Tech Solutions Inc.", fill='black')
    d.text((20, 110), "Phone: +1 (555) 123-4567", fill='black')
    d.text((20, 140), "Email: john@techsolutions.com", fill='black')
    d.text((20, 170), "Website: www.techsolutions.com", fill='black')
    
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    buf.seek(0)
    return buf

def main():
    st.title("📇 Business Card Scanner")
    st.markdown("Upload business card images to extract contact information")
    
    # Initialize OCR
    reader = init_ocr()
    
    if reader is None:
        st.error("OCR engine failed to initialize. Please check the logs.")
        return
    
    # Settings
    auto_save = st.sidebar.checkbox("Auto-save to CSV", value=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Upload Business Card")
        uploaded_file = st.file_uploader(
            "Choose an image file", 
            type=['png', 'jpg', 'jpeg'],
            help="Upload a clear photo of a business card"
        )
        
        st.markdown("---")
        if st.button("Use Sample Card for Testing"):
            uploaded_file = create_sample_card()
    
    with col2:
        st.subheader("Extracted Information")
        
        previous_scans = load_scans()
        
        if uploaded_file is not None:
            # Display image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Card", use_column_width=True)
            
            # Process image
            with st.spinner("Scanning card..."):
                try:
                    # Convert to numpy array for EasyOCR
                    img_array = np.array(image)
                    
                    # Run OCR
                    results = reader.readtext(img_array, detail=0, paragraph=True)
                    text = " ".join(results) if results else ""
                    
                    if len(text) < MIN_TEXT_LENGTH:
                        st.error("Not enough text detected. Please try a clearer image.")
                    else:
                        # Extract fields
                        data = extract_fields_from_text(text)
                        
                        st.success("✅ Card scanned successfully!")
                        
                        # Display results
                        st.markdown("### Contact Details")
                        
                        if data['Name']:
                            st.text_input("Name", data['Name'], key="name")
                        if data['Company']:
                            st.text_input("Company", data['Company'], key="company")
                        if data['Role']:
                            st.text_input("Role", data['Role'], key="role")
                        if data['Phone']:
                            st.text_input("Phone", data['Phone'], key="phone")
                        if data['Email']:
                            st.text_input("Email", data['Email'], key="email")
                        if data['Website']:
                            st.text_input("Website", data['Website'], key="website")
                        
                        # Save if enabled
                        if auto_save:
                            if save_scan(data):
                                st.success("💾 Saved to scans.csv")
                        
                        # Show raw text
                        with st.expander("Show raw OCR text"):
                            st.text_area("Raw Text", text, height=150)
                            
                except Exception as e:
                    st.error(f"Error processing image: {str(e)}")
    
    # Display history
    st.markdown("---")
    st.subheader("Scan History")
    
    all_scans = load_scans()
    if all_scans:
        df = pd.DataFrame(all_scans)
        display_cols = [col for col in ['Name', 'Company', 'Role', 'Phone', 'Email', 'Timestamp'] 
                       if col in df.columns]
        st.dataframe(df[display_cols].sort_values('Timestamp', ascending=False), 
                    use_container_width=True)
        
        # Download button
        csv = df.to_csv(index=False)
        st.download_button(
            "Download CSV",
            data=csv,
            file_name="business_cards.csv",
            mime="text/csv"
        )
    else:
        st.info("No scans yet. Upload a business card to get started!")

if __name__ == "__main__":
    main()
