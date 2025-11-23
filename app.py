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
import threading
from typing import List, Dict, Any

# Try to import PaddleOCR (optional)
try:
    from paddleocr import PaddleOCR
    PADDLE_AVAILABLE = True
except ImportError:
    PADDLE_AVAILABLE = False
    st.warning("PaddleOCR not available. Using EasyOCR only.")

# Try to import streamlit-webrtc for live camera
try:
    from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration, WebRtcMode
    WEBRTC_AVAILABLE = True
except ImportError:
    WEBRTC_AVAILABLE = False

# Page config
st.set_page_config(
    page_title="Pro Business Card Scanner",
    page_icon="📇",
    layout="wide"
)

# Configuration
DATA_FILE = "scans.csv"
MIN_TEXT_LENGTH = 10
OCR_EVERY_N_FRAMES = 30

# Regex patterns for field extraction
PHONE_REGEX = re.compile(r'(\+?\d[\d\s\-\(\)]{7,}\d)')
TOLLFREE_REGEX = re.compile(r'\b(1?[-\s]?(800|888|877|866|855)[-\s]?\d{3}[-\s]?\d{4})\b')
EMAIL_REGEX = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
WEBSITE_REGEX = re.compile(r'(https?://[^\s]+|www\.[^\s]+|\b[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b)')

# Global state for live camera results
if 'live_scans' not in st.session_state:
    st.session_state.live_scans = []
if 'camera_active' not in st.session_state:
    st.session_state.camera_active = False

@st.cache_resource
def init_ocr():
    """Initialize OCR readers - cached to load only once"""
    st.info("🔄 Loading OCR models... This may take a minute.")
    
    # Initialize EasyOCR
    try:
        easy_reader = easyocr.Reader(['en'])
        st.success("✅ EasyOCR loaded successfully!")
    except Exception as e:
        st.error(f"❌ EasyOCR failed to load: {str(e)}")
        return None, None
    
    # Initialize PaddleOCR if available - FIXED: removed cls parameter
    paddle_reader = None
    if PADDLE_AVAILABLE:
        try:
            # Try different initialization methods for compatibility
            paddle_reader = PaddleOCR(use_angle_cls=False, lang='en')  # Removed cls parameter
            st.success("✅ PaddleOCR loaded successfully!")
        except Exception as e:
            st.warning(f"⚠️ PaddleOCR initialization failed: {str(e)}")
            try:
                # Try without use_angle_cls
                paddle_reader = PaddleOCR(lang='en')
                st.success("✅ PaddleOCR loaded successfully (without angle cls)!")
            except Exception as e2:
                st.warning(f"⚠️ PaddleOCR fallback also failed: {str(e2)}")
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
        easy_results = easy_reader.readtext(image, detail=0, paragraph=False)
        easy_text = " ".join(easy_results)
        all_text += easy_text + " "
    except Exception as e:
        st.warning(f"EasyOCR error: {str(e)}")
    
    # PaddleOCR (if available) - FIXED: handle different PaddleOCR versions
    if paddle_reader:
        try:
            # Convert to BGR for PaddleOCR
            image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            # Try different method calls for compatibility
            try:
                paddle_results = paddle_reader.ocr(image_bgr, cls=True)
            except TypeError:
                # If cls parameter not supported, try without it
                paddle_results = paddle_reader.ocr(image_bgr)
            
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
    # Ensure text is not None
    if text is None:
        text = ""
        
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    
    data = {
        'Name': '',
        'Company': '',
        'Role': '',
        'Phone': '',
        'TollFree': '',
        'Email': '',
        'Website': '',
        'RawText': text,
        'Source': 'upload'
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
        data['TollFree'] = tollfree[0][0]
    
    # Extract websites
    websites = WEBSITE_REGEX.findall(text)
    if websites:
        for web in websites:
            if not any(email in web for email in emails if emails):
                data['Website'] = web
                break
    
    # Name, Company, Role extraction heuristics
    if lines:
        data['Name'] = lines[0]
        
        # Look for company indicators
        for i, line in enumerate(lines[1:4]):
            line_upper = line.upper()
            if any(indicator in line_upper for indicator in ['INC', 'CORP', 'COMPANY', 'LTD', 'CO.', 'LLC']):
                data['Company'] = line
                break
            elif line_upper == line and len(line.split()) <= 4:
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
    FIXED: Handle None values properly
    """
    if not existing_data:
        return False
    
    new_text = new_data.get('RawText', '')
    if new_text is None:
        new_text = ''
    
    for existing in existing_data:
        existing_text = existing.get('RawText', '')
        
        # FIX: Handle None existing_text
        if existing_text is None:
            existing_text = ''
        
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
            df = pd.read_csv(DATA_FILE)
            # FIX: Replace NaN values with empty strings
            df = df.fillna('')
            return df.to_dict('records')
        except Exception as e:
            st.warning(f"Error loading previous scans: {e}")
            return []
    return []

def save_scan(data):
    """Save scan to CSV"""
    try:
        df = pd.DataFrame([data])
        
        if os.path.exists(DATA_FILE):
            existing_df = pd.read_csv(DATA_FILE)
            existing_df = existing_df.fillna('')  # Clean existing data
            df = pd.concat([existing_df, df], ignore_index=True)
        
        df.to_csv(DATA_FILE, index=False)
        return True
    except Exception as e:
        st.error(f"Error saving scan: {e}")
        return False

def process_image(image, easy_reader, paddle_reader, auto_crop=True, source="upload"):
    """Process image and extract business card data"""
    try:
        # Convert PIL to OpenCV if needed
        if isinstance(image, Image.Image):
            image_cv = np.array(image)
            image_cv = cv2.cvtColor(image_cv, cv2.COLOR_RGB2BGR)
        else:
            image_cv = image
        
        # Auto-crop if enabled
        if auto_crop:
            cropped_image, success = auto_crop_card(image_cv)
            if success:
                image_cv = cropped_image
                st.success("✅ Card auto-detected and cropped!")
            else:
                st.warning("⚠️ Could not auto-detect card. Using full image.")
        
        # Convert to RGB for display and OCR
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
        data['Source'] = source
        
        return data, image_rgb
        
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
        return None, None

# WebRTC Video Transformer for live camera
if WEBRTC_AVAILABLE:
    class BusinessCardTransformer(VideoTransformerBase):
        def __init__(self):
            self.frame_count = 0
            self.easy_reader = None
            self.paddle_reader = None
            self.auto_crop = True
            
        def set_readers(self, easy_reader, paddle_reader, auto_crop):
            self.easy_reader = easy_reader
            self.paddle_reader = paddle_reader
            self.auto_crop = auto_crop
            
        def transform(self, frame):
            img = frame.to_ndarray(format="bgr24")
            self.frame_count += 1
            
            # Only process every N frames to reduce load
            if self.frame_count % OCR_EVERY_N_FRAMES == 0 and self.easy_reader:
                try:
                    # Process the frame
                    data, processed_img = process_image(
                        img, self.easy_reader, self.paddle_reader, 
                        self.auto_crop, "camera"
                    )
                    
                    if data is not None:
                        # Check for duplicates before adding
                        previous_scans = load_scans()
                        current_session_scans = st.session_state.get('live_scans', [])
                        all_scans = previous_scans + current_session_scans
                        
                        if not is_duplicate(data, all_scans):
                            st.session_state.live_scans.append(data)
                            
                            # Auto-save if enabled
                            if st.session_state.get('auto_save', True):
                                save_scan(data)
                    
                    # Add overlay to show camera is active
                    h, w = img.shape[:2]
                    cv2.putText(img, "LIVE SCAN - Hold card steady", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(img, f"Frame: {self.frame_count}", (10, h - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                               
                except Exception as e:
                    # Don't crash on OCR errors
                    print(f"OCR error in camera: {e}")
            
            return img

def create_sample_card():
    """Create a sample business card image for testing"""
    from PIL import Image, ImageDraw, ImageFont
    
    # Create a blank image
    img = Image.new('RGB', (400, 250), color='white')
    d = ImageDraw.Draw(img)
    
    # Use default font
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

# Main app
def main():
    st.title("📇 Business Card Scanner")
    st.markdown("Upload images or use live camera to scan business cards")
    
    # Initialize OCR readers
    easy_reader, paddle_reader = init_ocr()
    
    if easy_reader is None:
        st.error("❌ OCR failed to initialize. Please check the requirements.")
        return
    
    if not PADDLE_AVAILABLE:
        st.info("ℹ️ Running with EasyOCR only.")
    
    # Sidebar
    st.sidebar.header("Settings")
    auto_crop = st.sidebar.checkbox("Auto-detect & crop card", value=True)
    auto_save = st.sidebar.checkbox("Auto-save to scans.csv", value=True)
    st.session_state.auto_save = auto_save
    
    # Main tabs
    tab1, tab2, tab3 = st.tabs(["📁 Upload Image", "📷 Live Camera", "📊 Scan History"])
    
    with tab1:
        st.subheader("Upload Business Card Image")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
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
                sample_image = create_sample_card()
                uploaded_file = sample_image
        
        with col2:
            st.subheader("Upload Results")
            
            # Load previous scans
            previous_scans = load_scans()
            
            if uploaded_file is not None:
                # Read image
                image = Image.open(uploaded_file)
                
                # Display original image
                st.image(image, caption="Uploaded Image", use_column_width=True)
                
                # Process image
                data, processed_image = process_image(
                    image, easy_reader, paddle_reader, auto_crop, "upload"
                )
                
                if data is not None and processed_image is not None:
                    # Display processed image if cropped
                    if auto_crop:
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
                            st.text_input("Name", data['Name'], key="name_up")
                        if data['Company']:
                            st.text_input("Company", data['Company'], key="company_up")
                        if data['Role']:
                            st.text_input("Role", data['Role'], key="role_up")
                    
                    with col2:
                        if data['Phone']:
                            st.text_input("Phone", data['Phone'], key="phone_up")
                        if data['Email']:
                            st.text_input("Email", data['Email'], key="email_up")
                        if data['Website']:
                            st.text_input("Website", data['Website'], key="website_up")
                    
                    # Save if not duplicate and auto-save enabled
                    if not is_duplicate(data, previous_scans) and auto_save:
                        if save_scan(data):
                            st.success("💾 Saved to scans.csv")
                    
                    # Show raw OCR text
                    with st.expander("Show raw OCR text"):
                        st.text_area("Raw Text", data['RawText'], height=150, key="raw_up")
    
    with tab2:
        st.subheader("Live Camera Scan")
        
        if WEBRTC_AVAILABLE and easy_reader:
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown("**Instructions:**")
                st.markdown("""
                1. Click **START CAMERA** below
                2. Allow camera permissions in your browser
                3. Hold the business card steady in front of the camera
                4. The app will automatically scan every few seconds
                5. Detected cards will appear in the table below
                """)
                
                # Start WebRTC camera
                rtc_config = RTCConfiguration({
                    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
                })
                
                webrtc_ctx = webrtc_streamer(
                    key="business-card-camera",
                    mode=WebRtcMode.SENDRECV,
                    rtc_configuration=rtc_config,
                    video_transformer_factory=BusinessCardTransformer,
                    media_stream_constraints={"video": True, "audio": False},
                    async_transform=True,
                )
                
                if webrtc_ctx.video_transformer:
                    webrtc_ctx.video_transformer.set_readers(easy_reader, paddle_reader, auto_crop)
            
            with col2:
                st.markdown("### Live Results")
                
                # Show live scan results
                if st.session_state.live_scans:
                    live_df = pd.DataFrame(st.session_state.live_scans)
                    display_cols = [col for col in ['Name', 'Company', 'Role', 'Phone', 'Email'] 
                                  if col in live_df.columns]
                    st.dataframe(live_df[display_cols].tail(5), use_container_width=True)
                    
                    if st.button("Clear Live Results"):
                        st.session_state.live_scans = []
                        st.rerun()
                else:
                    st.info("No live scans yet. Hold a card in front of the camera.")
        
        else:
            st.error("Live camera not available. Please install streamlit-webrtc or use image upload.")
    
    with tab3:
        st.subheader("All Scanned Cards")
        
        # Combine live scans and saved scans
        saved_scans = load_scans()
        all_scans = saved_scans + st.session_state.live_scans
        
        if all_scans:
            scans_df = pd.DataFrame(all_scans)
            
            # Remove RawText column for display
            display_cols = [col for col in scans_df.columns if col != 'RawText']
            if display_cols:
                display_df = scans_df[display_cols]
                st.dataframe(display_df.sort_values('Timestamp', ascending=False), 
                            use_container_width=True,
                            height=400)
                
                # Download button
                csv = scans_df.to_csv(index=False)
                st.download_button(
                    label="Download all scans as CSV",
                    data=csv,
                    file_name="business_cards.csv",
                    mime="text/csv"
                )
        else:
            st.info("No scans yet. Upload an image to get started!")

if __name__ == "__main__":
    main()
