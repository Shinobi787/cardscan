# Auto Business Card Scanner (EasyOCR)

This Streamlit app scans business cards using EasyOCR (no Google Cloud, no billing) and saves results into scans.csv.

## Files
- app.py
- requirements.txt

## How to run locally
1. Create virtualenv:
   python -m venv venv
   source venv/bin/activate   # macOS/Linux
   venv\Scripts\activate      # Windows

2. Install:
   pip install -r requirements.txt

3. Run:
   streamlit run app.py

## Deploy to Streamlit Cloud
1. Push repository to GitHub.
2. Create a new app on Streamlit Cloud and link the repo + branch.
3. Use the default start command (streamlit run app.py).
4. Allow camera permission in your browser. Click "Take photo" to capture the card (mobile browsers also work).
5. Scans are appended automatically if they are unique.

## Notes
- This app uses snapshot captures (browser 'Take photo'). True continuous streaming requires WebRTC and reliable TURN servers which are not recommended for Streamlit Cloud.
- If you want OCR quality improvements, I can add: perspective auto-crop, image enhancement, or a simple UI for selecting best OCR result per card.
