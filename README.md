# Business Card Scanner — Ensemble OCR (EasyOCR + PaddleOCR) with Card Detection

This project is a production-ready business card scanning system with:

- **Automatic card detection** (OpenCV contour detection + perspective transform)
- **Dual OCR engine (Ensemble)**  
  - EasyOCR (fast)  
  - PaddleOCR (accurate)  
- **Advanced preprocessing** (sharpening, contrast, denoising, resizing)
- **High-accuracy field extraction**  
  - Name  
  - Company  
  - Role  
  - Phone  
  - Toll-Free  
  - Email  
  - Website  
- **Auto-scan mode** (via Streamlit `camera_input`)
- **CSV Export**
- **Deduplication to avoid duplicate scans**
- **Test sample button** to run OCR on a known image

This is the most accurate FREE OCR pipeline you can run on Streamlit Cloud (no API keys, no billing).

---

## 🚀 Deployment on Streamlit Cloud

### **1. Add Runtime File**
You MUST force Python 3.10.  
Create this file in your repo:

