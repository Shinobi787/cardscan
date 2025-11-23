# Business Card Scanner (Streamlit)

## What this does
Captures a single business card image (camera or upload), pre-processes it, runs Tesseract OCR and extracts Name, Company, Role, Phone, Email, Website. Results can be edited and saved to scans.csv. You can download a CSV of all saved scans.

## Important: install Tesseract OCR (system-level)
pytesseract is a Python wrapper. You must install the Tesseract engine:

- Ubuntu/Debian:
  sudo apt-get update
  sudo apt-get install -y tesseract-ocr

- macOS (Homebrew):
  brew install tesseract

- Windows:
  Download and install from the tesseract project (or use Chocolatey).

## Run locally
1. Create and activate virtualenv:
   python -m venv venv
   source venv/bin/activate   # macOS/Linux
   venv\Scripts\activate      # Windows

2. Install Python deps:
   pip install -r requirements.txt

3. Run:
   streamlit run app.py

## Deploy to Streamlit Cloud
1. Push repo to GitHub.
2. Create new app on Streamlit Cloud referencing the repo.
3. **Caveat**: Streamlit Cloud may NOT include system packages like Tesseract. If it's missing, options:
   - Deploy using a Docker image where you install tesseract-ocr.
   - Deploy to a server (VPS) where you can apt-get install tesseract.
   - Replace OCR with a cloud OCR API (Google Vision, Azure, AWS) and call the API in `image_to_text()`; this avoids the need for local Tesseract.

## Notes & improvements
- Heuristics are simple — for best results use a layout-aware OCR or ML model.
- You can integrate with Google Sheets / Airtable / database for live sync.
