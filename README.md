# Pro Business Card Scanner — EasyOCR + PaddleOCR ensemble

A professional business card scanner that uses ensemble OCR (EasyOCR + PaddleOCR) for accurate text extraction with auto-cropping and advanced field parsing.

## Features

- **Auto-detection & Cropping**: Automatically detects and crops business cards from photos
- **Ensemble OCR**: Combines EasyOCR + PaddleOCR for maximum accuracy
- **Advanced Field Extraction**: Name, Company, Role, Phone, TollFree, Email, Website
- **Auto-deduplication**: Prevents duplicate entries
- **CSV Export**: Saves all scans to `scans.csv`
- **Sample Testing**: Built-in sample card for testing

## Installation & Run (Locally)

1. **Create virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate    # macOS/Linux
   venv\Scripts\activate       # Windows
