# Business Card Scanner — EasyOCR, No OpenCV (Cloud-friendly)

This app scans business cards using EasyOCR and improved image preprocessing (no OpenCV). It auto-detects unique scans and saves them to scans.csv.

## How it works
- Use the browser camera "Take photo" button or upload an image.
- The app preprocesses the image (contrast, sharpen, denoise, resize).
- EasyOCR reads text; the app then extracts Name/Company/Role/Phone/TollFree/Email/Website.
- Unique scans are appended to the session and saved to scans.csv (if auto-save enabled).
- There's a "Test with sample image" button that uses a sample file at `/mnt/data/07c1090d-a232-4e82-bb1f-16abd2b9ea93.png` (your uploaded image).

## Setup
1. Create a repo and add `app.py` and `requirements.txt`.
2. Deploy to Streamlit Cloud (or run locally).
   - Locally:
     python -m venv venv
     source venv/bin/activate   # macOS/Linux
     venv\\Scripts\\activate    # Windows
     pip install -r requirements.txt
     streamlit run app.py

## Tips for best results
- Good lighting and a flat, centered card give the best OCR results.
- If OCR misses fields, try "Show raw OCR text" to see what the OCR read and adjust lighting/position.
- If you want even better accuracy, we can add an optional PaddleOCR or ONNX-based crop/detector — but those may require additional installations.

