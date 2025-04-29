# main.py

# --- Imports ---
import pytesseract
import cv2
import os
import numpy as np
import json
import re
from collections import defaultdict
import statistics # Although not used in the simpler extraction, keep if needed later
import tempfile # For handling uploaded files
import shutil   # For copying file objects
from fastapi import FastAPI, File, UploadFile, HTTPException
from typing import List, Dict, Any, Optional

# --- Tesseract Configuration ---
# ===> IMPORTANT: VERIFY THIS PATH IS CORRECT FOR YOUR SYSTEM <===
try:
    # Common locations - MODIFY OR ADD YOURS
    tesseract_path_options = [
        r'C:\Program Files\Tesseract-OCR\tesseract.exe',
    ]
    tesseract_exe_path = None
    for path in tesseract_path_options:
        if os.path.exists(path):
            tesseract_exe_path = path
            break

    if tesseract_exe_path:
        print(f"Setting Tesseract executable path to: {tesseract_exe_path}")
        pytesseract.pytesseract.tesseract_cmd = tesseract_exe_path
    else:
        print("ERROR: Tesseract executable not found in common paths.")
        print("Please set the CORRECT path manually for 'pytesseract.pytesseract.tesseract_cmd'")
        raise pytesseract.TesseractNotFoundError()

except pytesseract.TesseractNotFoundError:
    print("*"*80)
    print("FATAL ERROR: Tesseract not found or path not set correctly in the script.")
    print("Script cannot run without Tesseract.")
    print("*"*80)
    # In a real deployment, you might raise an exception or log differently
    # For now, we'll let it potentially fail later if called without Tesseract
    pass # Allow script load, but OCR will fail
except Exception as e:
    print(f"An unexpected error occurred during Tesseract path configuration: {e}")
    # Decide if you want to exit or continue
    pass


# --- Configuration for Extraction Logic (from second script) ---
Y_TOLERANCE = 10
X_TOLERANCE = 40
MIN_CONFIDENCE = 60 # Used in extraction filtering

# --- Regular Expressions & Keywords (from second script) ---
REGEX_VALUE = re.compile(r"^(?:[<>]\s?)?\d+(\.\d+)?$")
REGEX_RANGE = re.compile(
    r"^\d+(\.\d+)?\s*-\s*\d+(\.\d+)?$"
    r"|^<\s*\d+(\.\d+)?"
    r"|^>\s*\d+(\.\d+)?"
    r"|^UP TO\s+\d+(\.\d+)?"
    r"|^\d+\s*-\s*\d+$"
)
KNOWN_UNITS = {"%", "g/dl", "mg/dl", "iu/l", "u/l", "/ul", "x10^3/ul",
               "cells/ul", "fl", "pg", "gm/dl", "mm/hr", "me/l", "ng/"}
# Use lowercase set for efficiency
FILTER_KEYWORDS = {kw.lower() for kw in {
    'report', 'patient', 'doctor', 'hospital', 'clinic', 'date',
    'sample', 'serum', 'blood', 'specimen', 'page', 'regd', 'lab',
    'signature', 'technician', 'authorized', 'printed', 'cygnus',
    'sterling', 'ruby', 'hall', 'good', 'life', 'oscar', 'diagnostic',
    'superspeciality', 'medical', 'foundation', 'test', 'result',
    'unit', 'range', 'investigation', 'biochemistry', 'haematology',
    'header', 'footer', 'department', 'method', 'parameter', 'sr no',
    'uhid', 'ip no', 'ref by', 'end of the report'
}}

# --- FastAPI App Initialization ---
app = FastAPI(
    title="Lab Report OCR Service",
    description="Accepts a lab report image and extracts test data using Tesseract OCR and rule-based parsing.",
    version="1.0.0"
)

# === OCR Function (Step 1 Logic - with basic preprocessing) ===
def preprocess_and_ocr_tesseract(image_path):
    """
    Loads an image, performs OCR using Tesseract with preprocessing, and returns structured results.
    Returns None on failure.
    """
    # Basic check included in the function
    # if not os.path.exists(image_path): return None

    try:
        image = cv2.imread(image_path)
        if image is None:
             print(f"Error: cv2.imread failed for {image_path}")
             return None

        # === Preprocessing ===
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Using adaptive thresholding might be better sometimes, but Otsu is simpler
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Perform OCR
        print(f"Processing image with Tesseract: {image_path}")
        # Use default PSM 3 (fully automatic page segmentation) unless specified otherwise
        ocr_result = pytesseract.image_to_data(thresh, output_type=pytesseract.Output.DICT, lang='eng')

        if not ocr_result or not ocr_result.get('text'):
            print("Warning: Tesseract returned empty result dict or no text.")
            return []

        extracted_data = []
        num_words = len(ocr_result['text'])
        for i in range(num_words):
             # Use .get with default to avoid KeyError if Tesseract output is malformed
            conf = int(ocr_result.get('conf', [-1])[i]) # Default confidence -1
            text = ocr_result.get('text', [''])[i].strip()

            if conf >= 0 and text: # Check confidence >= 0 and non-empty text
                x = ocr_result.get('left', [0])[i]
                y = ocr_result.get('top', [0])[i]
                w = ocr_result.get('width', [0])[i]
                h = ocr_result.get('height', [0])[i]

                if w > 0 and h > 0: # Basic check for valid box dimensions
                    box = [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]
                    extracted_data.append({
                        "text": text,
                        "confidence": float(conf),
                        "box": box
                    })

        print(f"OCR complete. Detected {len(extracted_data)} text blocks.")
        return extracted_data

    except pytesseract.TesseractNotFoundError:
         print("Tesseract Error: Executable not found. Check path configuration.")
         # Re-raise or handle as needed for the API
         raise HTTPException(status_code=500, detail="Tesseract configuration error on server.")
    except Exception as e:
        print(f"Error during OCR processing for {image_path}: {e}")
        # Raise HTTPException for FastAPI to return a server error
        raise HTTPException(status_code=500, detail=f"Internal server error during OCR: {e}")


# === Extraction Functions (Step 2 Logic - simpler version) ===

# --- Helper Functions (Copied from second script) ---
def get_box_center(box):
    x_coords = [p[0] for p in box]; y_coords = [p[1] for p in box]
    return sum(x_coords) / 4, sum(y_coords) / 4

def get_box_y_range(box):
    y_coords = [p[1] for p in box]; return min(y_coords), max(y_coords)

def get_box_x_start(box): return box[0][0]

def parse_numeric_value(value_str):
    match = re.search(r"(\d+(\.\d+)?)", value_str)
    try: return float(match.group(1)) if match else None
    except: return None

def parse_range(range_str):
    range_str = range_str.strip().upper().replace("UP TO", "<")
    low, high = None, None
    try:
        match = re.match(r"^\s*(\d+(\.\d+)?)\s*-\s*(\d+(\.\d+)?)\s*$", range_str)
        if match: low, high = float(match.group(1)), float(match.group(3)); return low, high
        match = re.match(r"^\s*<\s*(\d+(\.\d+)?)\s*$", range_str)
        if match: high = float(match.group(1)); return None, high
        match = re.match(r"^\s*>\s*(\d+(\.\d+)?)\s*$", range_str)
        if match: low = float(match.group(1)); return low, None
        match = re.match(r"^\s*(\d+)\s*-\s*(\d+)\s*$", range_str)
        if match: low, high = float(match.group(1)), float(match.group(2)); return low, high
    except (ValueError, TypeError): pass
    return None, None

def is_out_of_range(value_str, range_str):
    value = parse_numeric_value(value_str)
    if value is None: return None
    low, high = parse_range(range_str)
    if low is not None and high is not None: return not (low <= value <= high)
    if high is not None: return value >= high
    if low is not None: return value <= low
    return None

# --- Main Extraction Function (Simpler Logic) ---
def extract_lab_data_from_ocr(ocr_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Processes OCR results using simpler rule-based logic assuming fixed column order.
    """
    if not ocr_results: return []

    # 1. Filter blocks
    filtered_blocks = []
    for block in ocr_results:
        if not all(k in block for k in ['text', 'confidence', 'box']): continue
        text_lower = block['text'].lower().strip()
        # Use MIN_CONFIDENCE from config
        if block['confidence'] < MIN_CONFIDENCE or not text_lower: continue

        # Use lowercase FILTER_KEYWORDS set for efficient check
        is_noise = any(keyword in text_lower for keyword in FILTER_KEYWORDS if len(keyword)>1)

        # Short irrelevant check
        is_short_irrelevant = len(text_lower) <= 1 and not text_lower.isdigit() and text_lower not in KNOWN_UNITS

        if not is_noise and not is_short_irrelevant:
            filtered_blocks.append(block)

    if not filtered_blocks: return []

    # 2. Group into Lines (Simple Y-Coord Grouping)
    filtered_blocks.sort(key=lambda b: get_box_y_range(b['box'])[0])
    lines = []
    current_line = []
    if filtered_blocks:
        current_line.append(filtered_blocks[0])
        last_y_min, last_y_max = get_box_y_range(filtered_blocks[0]['box'])
        for i in range(1, len(filtered_blocks)):
            block = filtered_blocks[i]
            y_min, y_max = get_box_y_range(block['box'])
            # Grouping logic using Y_TOLERANCE
            if abs(y_min - last_y_min) < Y_TOLERANCE or abs(y_max - last_y_max) < Y_TOLERANCE or \
               (y_min >= last_y_min - Y_TOLERANCE and y_min <= last_y_max + Y_TOLERANCE):
                current_line.append(block)
                last_y_min = min(last_y_min, y_min); last_y_max = max(last_y_max, y_max)
            else:
                if current_line: lines.append(sorted(current_line, key=lambda b: get_box_x_start(b['box'])))
                current_line = [block]
                last_y_min, last_y_max = y_min, y_max
        if current_line: lines.append(sorted(current_line, key=lambda b: get_box_x_start(b['box'])))

    # 3. Process Lines (Assuming Name -> Value -> Unit -> Range order)
    extracted_tests = []
    for line in lines:
        if len(line) < 2: continue

        possible_entities = []
        for block in line:
            text = block['text'].strip()
            if not text: continue
            if REGEX_RANGE.match(text): possible_entities.append(("RANGE", text, block))
            elif REGEX_VALUE.match(text): possible_entities.append(("VALUE", text, block))
            elif text.lower() in KNOWN_UNITS or ('/' in text and len(text) < 10): possible_entities.append(("UNIT", text, block))
            else: possible_entities.append(("NAME_PART", text, block))

        # Assemble based on assumed order
        found_name = []; found_value = None; found_unit = None; found_range = None
        name_indices = [i for i, (type, _, _) in enumerate(possible_entities) if type == "NAME_PART"]
        if name_indices:
            consecutive_name_parts = []
            last_index = -1
            for i, index in enumerate(name_indices):
                if index == 0 or index == last_index + 1:
                    consecutive_name_parts.append(possible_entities[index][1])
                    last_index = index
                else: break
            if consecutive_name_parts: found_name = [" ".join(consecutive_name_parts)]

        for type, text, block in possible_entities:
            if type == "VALUE" and not found_value: found_value = text
            elif type == "UNIT" and not found_unit:
                 if found_value is None or text != found_value: found_unit = text
            elif type == "RANGE" and not found_range: found_range = text

        # Create result if Name, Value, and Range found
        if found_name and found_value and found_range:
            test_name = found_name[0]
            test_value = found_value
            test_unit = found_unit if found_unit else ""
            bio_ref_range = found_range
            if len(test_name) > 1 and not re.match(r"^[0-9\s.<>-]+$", test_name) and test_value != bio_ref_range:
                out_of_range = is_out_of_range(test_value, bio_ref_range)
                extracted_tests.append({
                    "test_name": test_name.strip(),
                    "test_value": test_value.strip(),
                    "bio_reference_range": bio_ref_range.strip(),
                    "test_unit": test_unit.strip(),
                    "lab_test_out_of_range": out_of_range
                })
    print(f"Extraction complete. Found {len(extracted_tests)} potential lab tests.")
    return extracted_tests


# === FastAPI Endpoint ===

@app.post("/get-lab-tests/")
async def get_lab_tests(image: UploadFile = File(...)):
    """
    Accepts an image file, performs OCR, extracts lab test data,
    and returns the results in JSON format.
    """
    tmp_path = None
    try:
        # Validate file type (optional but recommended)
        if not image.content_type.startswith("image/"):
             raise HTTPException(status_code=400, detail="Invalid file type. Please upload an image.")

        # Create a temporary file to store the uploaded image
        # Use a context manager for cleaner handling if possible, but need path outside
        suffix = os.path.splitext(image.filename)[1] or ".png" # Keep original suffix or default
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            shutil.copyfileobj(image.file, tmp)
            tmp_path = tmp.name # Get the path of the temp file
        print(f"Image saved temporarily to: {tmp_path}")

        # --- Run OCR (Step 1) ---
        print("Starting OCR step...")
        ocr_results = preprocess_and_ocr_tesseract(tmp_path)

        # Handle OCR failure or no results
        if ocr_results is None:
            # Exception was already raised by the function for internal errors
            # If it returns None due to file read error (handled inside now)
            raise HTTPException(status_code=500, detail="Failed to process image for OCR.")
        if not ocr_results:
             print("OCR yielded no results.")
             return {"is_success": False, "data": [], "error": "OCR yielded no text"}

        # --- Run Extraction (Step 2) ---
        print("Starting extraction step...")
        final_data = extract_lab_data_from_ocr(ocr_results)

        # --- Format Response ---
        if final_data:
            print("Extraction successful.")
            return {"is_success": True, "data": final_data}
        else:
            print("Extraction yielded no structured data.")
            return {"is_success": False, "data": [], "error": "Could not extract structured data from OCR results"}

    except HTTPException as http_exc:
         # Re-raise HTTP exceptions directly
         raise http_exc
    except Exception as e:
        print(f"An unexpected error occurred in the API endpoint: {e}")
        # Return a generic server error
        raise HTTPException(status_code=500, detail=f"An internal server error occurred: {e}")

    finally:
        # --- Cleanup: Ensure temporary file is deleted ---
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
                print(f"Temporary file deleted: {tmp_path}")
            except Exception as e:
                print(f"Error deleting temporary file {tmp_path}: {e}")


# --- Root Endpoint (Optional: for basic check) ---
@app.get("/")
async def read_root():
    return {"message": "Lab Report OCR Service is running. Use the /get-lab-tests/ endpoint to process images."}


# --- To Run (using uvicorn) ---
# Save this file as main.py
# Run in terminal: uvicorn main:app --reload
# Access API docs at: http://127.0.0.1:8000/docs