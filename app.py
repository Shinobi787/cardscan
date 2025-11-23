import streamlit as st
import cv2
import numpy as np
from PIL import Image
import pandas as pd
import re
import easyocr
import os
from datetime import datetime
import tempfile
import io

# Try to import PaddleOCR (optional)
try:
    from paddleocr import PaddleOCR
    PADDLE_AVAILABLE = True
except ImportError:
    PADDLE_AVAILABLE = False
    st.warning("PaddleOCR not available. Using EasyOCR only.")

# Page config
st.set_page_config(
    page_title="Pro Business Card Scanner",
    page_icon="📇",
    layout="wide"
)

# Configuration
DATA_FILE = "scans.csv"
MIN_TEXT_LENGTH = 10

# Regex patterns for field extraction
PHONE_REGEX = re.compile(r'(\+?\d[\d\s\-\(\)]{7,}\d)')
TOLLFREE_REGEX = re.compile(r'\b(1?[-\s]?(800|888|877|866|855)[-\s]?\d{3}[-\s]?\d{4})\b')
EMAIL_REGEX = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
WEBSITE_REGEX = re.compile(r'(https?://[^\s]+|www\.[^\s]+|\b[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b)')

def init_ocr():
    """Initialize OCR readers"""
    # Initialize EasyOCR
    easy_reader = easyocr.Reader(['en'])
    
    # Initialize PaddleOCR if available
    paddle_reader = None
    if PADDLE_AVAILABLE:
        try:
            paddle_reader = PaddleOCR(use_angle_cls=True, lang='en')
            st.success("PaddleOCR initialized successfully!")
        except Exception as e:
            st.warning(f"PaddleOCR initialization failed: {str(e)}")
            paddle_reader = None
    
    return easy_reader, paddle_reader

def auto_crop_card(image):
    """
    Auto-detect and crop business card from image
    Returns cropped image and success status
    """
    try:
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Edge detection
        edges = cv2.Canny(blurred, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return image, False
        
        # Find the largest contour (likely the card)
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Approximate the contour to a polygon
        epsilon = 0.02 * cv2.arcLength(largest_contour, True)
        approx = cv2.approxPolyDP(largest_contour, epsilon, True)
        
        # If we have a quadrilateral
        if len(approx) == 4:
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(largest_contour)
            
            # Add some padding
            padding = 10
            x = max(0, x - padding)
            y = max(0, y - padding)
            w = min(image.shape[1] - x, w + 2 * padding)
            h = min(image.shape[0] - y, h + 2 * padding)
            
            # Crop the image
            cropped = image[y:y+h, x:x+w]
            return cropped, True
        else:
            return image, False
            
    except Exception as e:
        st.error(f"Error in auto-crop: {str(e)}")
        return image, False

def ensemble_ocr(image, easy_reader, paddle_reader):
    """
    Run OCR using both EasyOCR and PaddleOCR (if available)
    Returns combined text
    """
    all_text = ""
    
    # EasyOCR
    try:
        easy_results = easy_reader.readtext(image, detail=0)
        easy_text = " ".join(easy_results)
        all_text += easy_text + " "
    except Exception as e:
        st.warning(f"EasyOCR error: {str(e)}")
    
    # PaddleOCR (if available)
    if paddle_reader:
        try:
            # Convert to BGR for PaddleOCR
            image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            paddle_results = paddle_reader.ocr(image_bgr, cls=True)
            
            if paddle_results and paddle_results[0]:
                paddle_texts = []
                for line in paddle_results[0]:
                    if line and len(line) >= 2:
                        paddle_texts.append(line[1][0])
                paddle_text = " ".join(paddle_texts)
                all_text += paddle_text + " "
        except Exception as e:
            st.warning(f"PaddleOCR error: {str(e)}")
    
    return all_text.strip()

def extract_fields_from_text(text):
    """
    Extract business card fields from OCR text
    """
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    
    data = {
        'Name': '',
        'Company': '',
        'Role': '',
        'Phone': '',
        'TollFree': '',
        'Email': '',
        'Website': '',
        'RawText': text
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
        # Take the longest phone number (most complete)
        data['Phone'] = max(phones, key=len)
    
    # Extract toll-free numbers
    tollfree = TOLLFREE_REGEX.findall(text)
    if tollfree:
        data['TollFree'] = tollfree[0][0]  # First group contains full number
    
    # Extract websites
    websites = WEBSITE_REGEX.findall(text)
    if websites:
        for web in websites:
            if not any(email in web for email in emails):  # Avoid email-like patterns
                data['Website'] = web
                break
    
    # Name, Company, Role extraction heuristics
    if lines:
        data['Name'] = lines[0]  # First line is often name
        
        # Look for company indicators
        for i, line in enumerate(lines[1:4]):  # Check next few lines
            line_upper = line.upper()
            if any(indicator in line_upper for indicator in ['INC', 'CORP', 'COMPANY', 'LTD', 'CO.', 'LLC']):
                data['Company'] = line
                break
            elif line_upper == line and len(line.split()) <= 4:  # All caps short line
                data['Company'] = line
                break
        
        # Look for role indicators
        role_indicators = ['manager', 'director', 'president', 'ceo', 'cto', 'cfo', 'engineer', 
                          'analyst', 'specialist', 'consultant', 'officer', 'head', 'lead']
        for line in lines[1:5]:
            line_lower = line.lower()
            if any(role in line_lower for role in role_indicators):
                data['Role'] = line
                break
    
    return data

def is_duplicate(new_data, existing_data, threshold=0.8):
    """
    Check if new scan is duplicate of existing ones
    """
    if not existing_data:
        return False
    
    new_text = new_data.get('RawText', '')
    for existing in existing_data:
        existing_text = existing.get('RawText', '')
        
        # Simple similarity check
        new_words = set(new_text.lower().split())
        existing_words = set(existing_text.lower().split())
        
        if len(new_words) == 0 or len(existing_words) == 0:
            continue
            
        intersection = new_words.intersection(existing_words)
        union = new_words.union(existing_words)
        
        similarity = len(intersection) / len(union) if union else 0
        
        if similarity > threshold:
            return True
    
    return False

def load_scans():
    """Load previous scans from CSV"""
    if os.path.exists(DATA_FILE):
        try:
            return pd.read_csv(DATA_FILE).to_dict('records')
        except:
            return []
    return []

def save_scan(data):
    """Save scan to CSV"""
    df = pd.DataFrame([data])
    
    if os.path.exists(DATA_FILE):
        existing_df = pd.read_csv(DATA_FILE)
        df = pd.concat([existing_df, df], ignore_index=True)
    
    df.to_csv(DATA_FILE, index=False)

def process_image(image, easy_reader, paddle_reader, auto_crop=True):
    """Process image and extract business card data"""
    # Convert PIL to OpenCV
    image_cv = np.array(image)
    image_cv = cv2.cvtColor(image_cv, cv2.COLOR_RGB2BGR)
    
    # Auto-crop if enabled
    if auto_crop:
        cropped_image, success = auto_crop_card(image_cv)
        if success:
            image_cv = cropped_image
            st.success("✅ Card auto-detected and cropped!")
        else:
            st.warning("⚠️ Could not auto-detect card. Using full image.")
    
    # Convert back to RGB for display
    image_rgb = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
    
    # Run ensemble OCR
    with st.spinner("Running OCR..."):
        text = ensemble_ocr(image_cv, easy_reader, paddle_reader)
    
    if len(text) < MIN_TEXT_LENGTH:
        st.error("❌ Not enough text detected. Please try a clearer image.")
        return None, image_rgb
    
    # Extract fields
    data = extract_fields_from_text(text)
    data['Timestamp'] = datetime.now().isoformat()
    
    return data, image_rgb

# Main app
def main():
    st.title("📇 Pro Business Card Scanner")
    st.markdown("EasyOCR + PaddleOCR ensemble with auto-cropping and advanced field extraction")
    
    # Initialize OCR readers
    easy_reader, paddle_reader = init_ocr()
    
    if not PADDLE_AVAILABLE:
        st.info("ℹ️ Running with EasyOCR only. For better accuracy, install PaddleOCR locally.")
    
    # Sidebar
    st.sidebar.header("Settings")
    auto_crop = st.sidebar.checkbox("Auto-detect & crop card", value=True)
    auto_save = st.sidebar.checkbox("Auto-save to scans.csv", value=True)
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Upload Business Card")
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Choose an image file", 
            type=['png', 'jpg', 'jpeg'],
            help="Upload a clear photo of a business card"
        )
        
        # Sample image for testing
        st.markdown("---")
        st.subheader("Test with Sample")
        if st.button("Use Sample Business Card"):
            # Create a sample business card image
            sample_image = create_sample_card()
            uploaded_file = sample_image
    
    with col2:
        st.subheader("Scan Results")
        
        # Load previous scans
        previous_scans = load_scans()
        
        if uploaded_file is not None:
            # Read image
            image = Image.open(uploaded_file)
            
            # Display original image
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            # Process image
            data, processed_image = process_image(
                image, easy_reader, paddle_reader, auto_crop
            )
            
            if data is not None:
                # Display processed image if cropped
                if auto_crop and not np.array_equal(np.array(image), processed_image):
                    st.image(processed_image, caption="Processed Image", use_column_width=True)
                
                # Check for duplicates
                if is_duplicate(data, previous_scans):
                    st.warning("⚠️ Similar card already scanned recently.")
                else:
                    st.success("✅ New card detected!")
                
                # Display extracted data
                st.markdown("### Extracted Information")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    if data['Name']:
                        st.text_input("Name", data['Name'])
                    if data['Company']:
                        st.text_input("Company", data['Company'])
                    if data['Role']:
                        st.text_input("Role", data['Role'])
                
                with col2:
                    if data['Phone']:
                        st.text_input("Phone", data['Phone'])
                    if data['TollFree']:
                        st.text_input("Toll Free", data['TollFree'])
                    if data['Email']:
                        st.text_input("Email", data['Email'])
                    if data['Website']:
                        st.text_input("Website", data['Website'])
                
                # Save if not duplicate and auto-save enabled
                if not is_duplicate(data, previous_scans) and auto_save:
                    save_scan(data)
                    st.success("💾 Saved to scans.csv")
                
                # Show raw OCR text
                with st.expander("Show raw OCR text"):
                    st.text_area("Raw Text", data['RawText'], height=150)
    
    # Display previous scans
    st.markdown("---")
    st.subheader("Previous Scans")
    
    if previous_scans:
        scans_df = pd.DataFrame(previous_scans)
        # Remove RawText column for display
        display_df = scans_df.drop(columns=['RawText'], errors='ignore')
        st.dataframe(display_df, use_container_width=True)
        
        # Download button
        csv = scans_df.to_csv(index=False)
        st.download_button(
            label="Download all scans as CSV",
            data=csv,
            file_name="business_cards.csv",
            mime="text/csv"
        )
    else:
        st.info("No scans yet. Upload a business card to get started!")

def create_sample_card():
    """Create a sample business card image for testing"""
    from PIL import Image, ImageDraw, ImageFont
    
    # Create a blank image
    img = Image.new('RGB', (400, 250), color='white')
    d = ImageDraw.Draw(img)
    
    # Use default font (size may vary by system)
    try:
        font_large = ImageFont.truetype("arial.ttf", 20)
        font_medium = ImageFont.truetype("arial.ttf", 14)
        font_small = ImageFont.truetype("arial.ttf", 12)
    except:
        font_large = font_medium = font_small = ImageFont.load_default()
    
    # Draw sample card content
    d.text((20, 20), "John Smith", fill='black', font=font_large)
    d.text((20, 50), "Marketing Director", fill='black', font=font_medium)
    d.text((20, 80), "Tech Solutions Inc.", fill='black', font=font_medium)
    d.text((20, 110), "Phone: +1 (555) 123-4567", fill='black', font=font_small)
    d.text((20, 130), "Email: john.smith@techsolutions.com", fill='black', font=font_small)
    d.text((20, 150), "Website: www.techsolutions.com", fill='black', font=font_small)
    
    # Convert to file-like object
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    buf.seek(0)
    
    return buf

if __name__ == "__main__":
    main()
