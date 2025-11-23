# Pro Business Card Scanner — EasyOCR + PaddleOCR ensemble

Features:
- Auto-detects and crops business card from photo.
- Ensemble OCR: EasyOCR + PaddleOCR (Paddle is optional; app falls back to EasyOCR).
- Advanced extraction logic: Name, Company, Role, Phone, TollFree, Email, Website.
- Auto-dedup and saves scans to scans.csv.
- Test with your uploaded sample image (button provided).

## Install & Run (locally)
1. Create environment:
   python -m venv venv
   source venv/bin/activate    # macOS/Linux
   venv\\Scripts\\activate     # Windows

2. Install:
   pip install -r requirements.txt

   NOTE: paddlepaddle may need a specific wheel for your OS. If installation fails, remove `paddleocr` and `paddlepaddle` from requirements and re-run pip install.

3. Run:
   streamlit run app.py

## Deploy to Streamlit Cloud
1. Push to GitHub.
2. Create a Streamlit Cloud app pointing to this repo.
3. Streamlit will install dependencies (may take longer because of easyocr/paddle).
4. If paddlepaddle installation fails in Cloud, the app will still run using EasyOCR only.

## Tips for best OCR
- Good lighting, focused, flat card yields best results.
- Use "Auto-detect & crop" to isolate the card before OCR (recommended).
- If detection fails on very cluttered backgrounds, try a clean background or use upload.

## Troubleshooting
- If PaddleOCR fails to initialize, you'll see a message and app will use EasyOCR only.
- If install for paddlepaddle times out on Streamlit Cloud, consider running locally or on a VPS (recommended for heavy workloads).
