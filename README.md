# Auto Business Card Scanner (Streamlit + Google Vision + WebRTC)

This app auto-scans business cards using the webcam and Google Vision OCR, then parses fields and stores entries.

## Setup (Google Cloud & secrets)

1. Create a Google Cloud Project.
2. Enable the Vision API for that project.
3. Create a Service Account (IAM) and add a JSON key.
4. In Streamlit Cloud, open your app → Settings → Secrets, and paste:

[google]
gcp_key = """
{ ... paste entire service account JSON file content here ... }
"""

(Use the triple-quotes exactly as above. Do NOT commit the JSON to GitHub.)

## Files
- app.py
- requirements.txt

## Run locally (optional)
1. Create virtualenv and install requirements:
   python -m venv venv
   source venv/bin/activate   # macOS/Linux
   venv\\Scripts\\activate    # Windows
   pip install -r requirements.txt

2. Run:
   streamlit run app.py

## Deploy to Streamlit Cloud
1. Push to GitHub.
2. Create a new app on Streamlit Cloud and link to the repo.
3. Add the secret (see above).
4. Start the app — the camera will ask for permission; hold the card in front of camera.

## Notes
- The app runs OCR every N frames (configurable in app). Adjust OCR_EVERY_N_FRAMES in `app.py`.
- The app stores detections in `scans.csv` if Auto-save is checked.
- If you want image auto-cropping (detect card box & perspective correction) or integration with Google Sheets / Airtable, I can add that next.
