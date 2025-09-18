# PyTesseract Documentation

PyTesseract is a Python wrapper for Google's Tesseract OCR (Optical Character Recognition) engine. It enables you to extract text from images and perform various OCR-related tasks programmatically.

## Table of Contents
- [Installation](#installation)
- [System Requirements](#system-requirements)
- [Configuration](#configuration)
- [Basic Usage](#basic-usage)
- [Core Functions](#core-functions)
- [Language Support](#language-support)
- [Image Preprocessing](#image-preprocessing)
- [Advanced Features](#advanced-features)
- [Output Formats](#output-formats)
- [Performance Optimization](#performance-optimization)
- [Error Handling](#error-handling)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)
- [API Reference](#api-reference)

## Installation

### Install PyTesseract
```bash
pip install pytesseract
```

### Install Pillow (Image Processing)
```bash
pip install Pillow
```

### Install OpenCV (Optional, for advanced preprocessing)
```bash
pip install opencv-python
```

## System Requirements

### Install Tesseract OCR Engine

#### Ubuntu/Debian
```bash
sudo apt update
sudo apt install tesseract-ocr
sudo apt install libtesseract-dev
```

#### macOS
```bash
# Using Homebrew
brew install tesseract

# Using MacPorts
sudo port install tesseract
```

#### Windows
1. Download installer from [GitHub Tesseract Releases](https://github.com/UB-Mannheim/tesseract/wiki)
2. Install to default location: `C:\Program Files\Tesseract-OCR\`
3. Add to PATH or configure manually in code

### Verify Installation
```bash
tesseract --version
```

## Configuration

### Basic Configuration
```python
import pytesseract
from PIL import Image

# Set tesseract executable path (if not in PATH)
# Windows
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# macOS (Homebrew)
pytesseract.pytesseract.tesseract_cmd = r'/opt/homebrew/bin/tesseract'

# macOS (MacPorts)
pytesseract.pytesseract.tesseract_cmd = r'/opt/local/bin/tesseract'

# Linux
pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'
```

### Advanced Configuration
```python
# Configure tessdata directory
pytesseract.pytesseract.tessdata_dir_config = r'--tessdata-dir "/opt/homebrew/share/tessdata"'

# Set timeout (in seconds)
pytesseract.pytesseract.timeout = 10

# Configure environment variables
import os
os.environ['TESSDATA_PREFIX'] = '/opt/homebrew/share/tessdata'
```

## Basic Usage

### Simple Text Extraction
```python
import pytesseract
from PIL import Image

# Load image
image = Image.open('sample.png')

# Extract text
text = pytesseract.image_to_string(image)
print(text)
```

### From Different Image Sources
```python
# From file path
text = pytesseract.image_to_string('image.png')

# From PIL Image
image = Image.open('image.png')
text = pytesseract.image_to_string(image)

# From numpy array (OpenCV)
import cv2
image = cv2.imread('image.png')
text = pytesseract.image_to_string(image)

# From URL
import requests
from io import BytesIO

response = requests.get('https://example.com/image.png')
image = Image.open(BytesIO(response.content))
text = pytesseract.image_to_string(image)
```

### Basic Configuration Options
```python
# Specify language
text = pytesseract.image_to_string(image, lang='eng')

# Custom configuration
custom_config = r'--oem 3 --psm 6'
text = pytesseract.image_to_string(image, config=custom_config)

# Combine language and config
text = pytesseract.image_to_string(image, lang='eng+fra', config=custom_config)
```

## Core Functions

### image_to_string()
Extract plain text from image.

```python
# Basic usage
text = pytesseract.image_to_string(image)

# With language
text = pytesseract.image_to_string(image, lang='eng')

# With custom config
text = pytesseract.image_to_string(image, config='--psm 6')

# With output type
text = pytesseract.image_to_string(image, output_type=pytesseract.Output.STRING)
```

### image_to_data()
Get detailed information including bounding boxes, confidence scores.

```python
# Get detailed data
data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)

# Access different components
words = data['text']
confidences = data['conf']
left = data['left']
top = data['top']
width = data['width']
height = data['height']

# Filter by confidence
threshold = 60
filtered_data = []
for i in range(len(data['text'])):
    if int(data['conf'][i]) > threshold:
        filtered_data.append({
            'text': data['text'][i],
            'confidence': data['conf'][i],
            'bbox': (data['left'][i], data['top'][i], 
                    data['width'][i], data['height'][i])
        })
```

### image_to_boxes()
Get character-level bounding boxes.

```python
# Get character boxes
boxes = pytesseract.image_to_boxes(image)

# Parse boxes
for line in boxes.splitlines():
    parts = line.split(' ')
    char = parts[0]
    x1, y1, x2, y2 = map(int, parts[1:5])
    print(f"Character '{char}' at ({x1},{y1}) to ({x2},{y2})")
```

### image_to_osd()
Get orientation and script detection information.

```python
# Get orientation and script detection
osd = pytesseract.image_to_osd(image)
print(osd)

# Parse OSD data
osd_data = pytesseract.image_to_osd(image, output_type=pytesseract.Output.DICT)
print(f"Orientation: {osd_data['orientation']}")
print(f"Rotate: {osd_data['rotate']}")
print(f"Orientation confidence: {osd_data['orientation_conf']}")
print(f"Script: {osd_data['script']}")
```

## Language Support

### Available Languages
```python
# Get list of available languages
languages = pytesseract.get_languages()
print(languages)
```

### Common Language Codes
```python
# Single language
text = pytesseract.image_to_string(image, lang='eng')  # English
text = pytesseract.image_to_string(image, lang='fra')  # French
text = pytesseract.image_to_string(image, lang='deu')  # German
text = pytesseract.image_to_string(image, lang='spa')  # Spanish
text = pytesseract.image_to_string(image, lang='chi_sim')  # Chinese Simplified
text = pytesseract.image_to_string(image, lang='jpn')  # Japanese
text = pytesseract.image_to_string(image, lang='ara')  # Arabic
text = pytesseract.image_to_string(image, lang='rus')  # Russian
text = pytesseract.image_to_string(image, lang='hin')  # Hindi

# Multiple languages
text = pytesseract.image_to_string(image, lang='eng+fra+deu')
```

### Installing Additional Languages
```bash
# Ubuntu/Debian
sudo apt install tesseract-ocr-fra  # French
sudo apt install tesseract-ocr-deu  # German
sudo apt install tesseract-ocr-spa  # Spanish

# macOS (Homebrew)
brew install tesseract-lang

# Download specific language data manually
wget https://github.com/tesseract-ocr/tessdata/raw/main/fra.traineddata
sudo mv fra.traineddata /usr/share/tesseract-ocr/4.00/tessdata/
```

## Image Preprocessing

### Using PIL for Preprocessing
```python
from PIL import Image, ImageEnhance, ImageFilter

def preprocess_image(image_path):
    # Load image
    image = Image.open(image_path)
    
    # Convert to grayscale
    image = image.convert('L')
    
    # Enhance contrast
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(2.0)
    
    # Enhance sharpness
    enhancer = ImageEnhance.Sharpness(image)
    image = enhancer.enhance(2.0)
    
    # Apply filters
    image = image.filter(ImageFilter.MedianFilter())
    
    return image

# Use preprocessed image
processed_image = preprocess_image('noisy_image.png')
text = pytesseract.image_to_string(processed_image)
```

### Using OpenCV for Advanced Preprocessing
```python
import cv2
import numpy as np

def advanced_preprocess(image_path):
    # Read image
    image = cv2.imread(image_path)
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Remove noise
    denoised = cv2.medianBlur(gray, 5)
    
    # Threshold the image
    _, thresh = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Morphological operations
    kernel = np.ones((1, 1), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # Dilation
    dilated = cv2.dilate(opening, kernel, iterations=1)
    
    return dilated

# Use advanced preprocessing
processed = advanced_preprocess('complex_image.png')
text = pytesseract.image_to_string(processed)
```

### Skew Correction
```python
import cv2
import numpy as np
from PIL import Image

def correct_skew(image):
    # Convert PIL to OpenCV
    if isinstance(image, Image.Image):
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply threshold
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Get the largest contour (assumed to be text)
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Get minimum area rectangle
    rect = cv2.minAreaRect(largest_contour)
    angle = rect[2]
    
    # Correct angle
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    
    # Rotate image
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    
    return rotated
```

## Advanced Features

### Page Segmentation Modes (PSM)
```python
# PSM modes
psm_modes = {
    0: 'Orientation and script detection (OSD) only',
    1: 'Automatic page segmentation with OSD',
    2: 'Automatic page segmentation, but no OSD, or OCR',
    3: 'Fully automatic page segmentation, but no OSD (Default)',
    4: 'Assume a single column of text of variable sizes',
    5: 'Assume a single uniform block of vertically aligned text',
    6: 'Assume a single uniform block of text',
    7: 'Treat the image as a single text line',
    8: 'Treat the image as a single word',
    9: 'Treat the image as a single word in a circle',
    10: 'Treat the image as a single character',
    11: 'Sparse text. Find as much text as possible in no particular order',
    12: 'Sparse text with OSD',
    13: 'Raw line. Treat the image as a single text line, bypassing hacks'
}

# Use specific PSM mode
text = pytesseract.image_to_string(image, config='--psm 6')
```

### OCR Engine Modes (OEM)
```python
# OEM modes
oem_modes = {
    0: 'Legacy engine only',
    1: 'Neural nets LSTM engine only',
    2: 'Legacy + LSTM engines',
    3: 'Default, based on what is available'
}

# Use specific OEM mode
text = pytesseract.image_to_string(image, config='--oem 1')
```

### Custom Configurations
```python
# Whitelist characters
config = '--psm 8 -c tessedit_char_whitelist=0123456789'
text = pytesseract.image_to_string(image, config=config)

# Blacklist characters
config = '--psm 6 -c tessedit_char_blacklist=!@#$%^&*()'
text = pytesseract.image_to_string(image, config=config)

# Multiple configurations
config = '--psm 6 --oem 1 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
text = pytesseract.image_to_string(image, config=config)
```

## Output Formats

### Different Output Types
```python
# String output (default)
text = pytesseract.image_to_string(image, output_type=pytesseract.Output.STRING)

# Bytes output
text_bytes = pytesseract.image_to_string(image, output_type=pytesseract.Output.BYTES)

# Dictionary output (for image_to_data)
data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)

# DataFrame output (requires pandas)
import pandas as pd
df = pytesseract.image_to_data(image, output_type=pytesseract.Output.DATAFRAME)
```

### Working with DataFrame Output
```python
import pandas as pd

# Get data as DataFrame
df = pytesseract.image_to_data(image, output_type=pytesseract.Output.DATAFRAME)

# Filter by confidence
high_conf = df[df['conf'] > 60]

# Filter by text (remove empty)
text_only = df[df['text'].str.strip() != '']

# Group by line
lines = df.groupby(['page_num', 'block_num', 'par_num', 'line_num'])['text'].apply(lambda x: ' '.join(x)).reset_index()

# Export to CSV
df.to_csv('ocr_results.csv', index=False)
```

## Performance Optimization

### Batch Processing
```python
import os
from concurrent.futures import ThreadPoolExecutor

def process_image(image_path):
    try:
        image = Image.open(image_path)
        text = pytesseract.image_to_string(image)
        return {'file': image_path, 'text': text, 'status': 'success'}
    except Exception as e:
        return {'file': image_path, 'text': '', 'status': f'error: {str(e)}'}

def batch_ocr(image_folder, max_workers=4):
    image_files = [f for f in os.listdir(image_folder) 
                  if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp'))]
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(process_image, 
                                  [os.path.join(image_folder, f) for f in image_files]))
    
    return results

# Process multiple images
results = batch_ocr('/path/to/images/', max_workers=8)
```

### Memory Optimization
```python
def memory_efficient_ocr(image_path):
    # Process image in chunks for large images
    image = Image.open(image_path)
    
    # Resize if too large
    max_size = 2000
    if max(image.size) > max_size:
        ratio = max_size / max(image.size)
        new_size = tuple(int(dim * ratio) for dim in image.size)
        image = image.resize(new_size, Image.Resampling.LANCZOS)
    
    # Process and immediately get text
    text = pytesseract.image_to_string(image)
    
    # Clean up
    image.close()
    
    return text
```

### Configuration Optimization
```python
# Fast configuration for simple text
fast_config = '--psm 6 --oem 3 -c tessedit_do_invert=0'

# Accurate configuration for complex layouts
accurate_config = '--psm 1 --oem 1'

# Number-only recognition
number_config = '--psm 8 --oem 3 -c tessedit_char_whitelist=0123456789'

def optimized_ocr(image, content_type='mixed'):
    configs = {
        'fast': '--psm 6 --oem 3',
        'accurate': '--psm 1 --oem 1',
        'numbers': '--psm 8 -c tessedit_char_whitelist=0123456789',
        'single_line': '--psm 7',
        'single_word': '--psm 8',
        'mixed': '--psm 3 --oem 3'
    }
    
    config = configs.get(content_type, configs['mixed'])
    return pytesseract.image_to_string(image, config=config)
```

## Error Handling

### Common Exceptions
```python
import pytesseract
from PIL import Image

def safe_ocr(image_path):
    try:
        # Check if file exists
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        # Load and process image
        image = Image.open(image_path)
        
        # Check if image is valid
        if image.size == (0, 0):
            raise ValueError("Invalid image size")
        
        # Perform OCR
        text = pytesseract.image_to_string(image)
        
        return {'success': True, 'text': text, 'error': None}
        
    except pytesseract.TesseractNotFoundError:
        return {'success': False, 'text': '', 
                'error': 'Tesseract not found. Please install Tesseract OCR.'}
    
    except pytesseract.TesseractError as e:
        return {'success': False, 'text': '', 
                'error': f'Tesseract error: {str(e)}'}
    
    except FileNotFoundError as e:
        return {'success': False, 'text': '', 
                'error': f'File error: {str(e)}'}
    
    except Exception as e:
        return {'success': False, 'text': '', 
                'error': f'Unexpected error: {str(e)}'}

# Usage
result = safe_ocr('image.png')
if result['success']:
    print(f"Extracted text: {result['text']}")
else:
    print(f"Error: {result['error']}")
```

### Timeout Handling
```python
import signal

class TimeoutError(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutError("OCR operation timed out")

def ocr_with_timeout(image, timeout_seconds=30):
    # Set timeout
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout_seconds)
    
    try:
        text = pytesseract.image_to_string(image)
        signal.alarm(0)  # Cancel timeout
        return text
    except TimeoutError:
        return "OCR operation timed out"
    except Exception as e:
        signal.alarm(0)  # Cancel timeout
        raise e
```

## Best Practices

### Image Quality Guidelines
```python
def check_image_quality(image_path):
    image = Image.open(image_path)
    
    checks = {
        'size_adequate': min(image.size) >= 300,  # Minimum 300px
        'aspect_ratio_reasonable': 0.1 <= image.size[0]/image.size[1] <= 10,
        'not_too_large': max(image.size) <= 4000,  # Maximum 4000px
        'format_supported': image.format in ['PNG', 'JPEG', 'TIFF', 'BMP']
    }
    
    return checks

# Validate before OCR
quality_checks = check_image_quality('image.png')
if all(quality_checks.values()):
    text = pytesseract.image_to_string(Image.open('image.png'))
else:
    print("Image quality issues:", quality_checks)
```

### Text Cleaning
```python
import re

def clean_ocr_text(text):
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove special characters if needed
    text = re.sub(r'[^\w\s\-.,!?;:]', '', text)
    
    # Fix common OCR mistakes
    replacements = {
        '0': 'O',  # Zero to O (context-dependent)
        '1': 'l',  # One to l (context-dependent)
        '5': 'S',  # Five to S (context-dependent)
    }
    
    # Apply replacements carefully
    # text = apply_smart_replacements(text, replacements)
    
    return text.strip()

def apply_smart_replacements(text, replacements):
    # Implement context-aware replacements
    # This is a simplified example
    words = text.split()
    corrected_words = []
    
    for word in words:
        # Apply replacements based on context
        # This would need more sophisticated logic
        corrected_words.append(word)
    
    return ' '.join(corrected_words)
```

### Confidence-Based Filtering
```python
def extract_high_confidence_text(image, min_confidence=60):
    data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
    
    high_conf_text = []
    for i in range(len(data['text'])):
        if int(data['conf'][i]) >= min_confidence:
            text = data['text'][i].strip()
            if text:  # Not empty
                high_conf_text.append({
                    'text': text,
                    'confidence': data['conf'][i],
                    'bbox': (data['left'][i], data['top'][i], 
                            data['width'][i], data['height'][i])
                })
    
    return high_conf_text
```

## Troubleshooting

### Common Issues and Solutions

#### Issue: "TesseractNotFoundError"
```python
# Solution 1: Set tesseract path
pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'

# Solution 2: Check installation
import subprocess
try:
    result = subprocess.run(['tesseract', '--version'], capture_output=True, text=True)
    print("Tesseract found:", result.stdout)
except FileNotFoundError:
    print("Tesseract not found. Please install Tesseract OCR.")
```

#### Issue: Poor OCR Accuracy
```python
# Solution: Preprocess image
def improve_accuracy(image_path):
    image = Image.open(image_path)
    
    # Convert to grayscale
    image = image.convert('L')
    
    # Increase size if too small
    if min(image.size) < 500:
        scale_factor = 500 / min(image.size)
        new_size = tuple(int(dim * scale_factor) for dim in image.size)
        image = image.resize(new_size, Image.Resampling.LANCZOS)
    
    # Enhance contrast
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(1.5)
    
    # Try different PSM modes
    psm_modes = [3, 6, 8, 7]
    results = []
    
    for psm in psm_modes:
        try:
            text = pytesseract.image_to_string(image, config=f'--psm {psm}')
            data = pytesseract.image_to_data(image, config=f'--psm {psm}', 
                                           output_type=pytesseract.Output.DICT)
            avg_conf = sum([int(c) for c in data['conf'] if int(c) > 0]) / len([c for c in data['conf'] if int(c) > 0])
            results.append({'psm': psm, 'text': text, 'confidence': avg_conf})
        except:
            continue
    
    # Return best result
    best_result = max(results, key=lambda x: x['confidence'])
    return best_result
```

#### Issue: Memory Issues with Large Images
```python
def process_large_image(image_path, chunk_size=1000):
    image = Image.open(image_path)
    
    # If image is too large, process in chunks
    if max(image.size) > chunk_size:
        # Resize maintaining aspect ratio
        ratio = chunk_size / max(image.size)
        new_size = tuple(int(dim * ratio) for dim in image.size)
        image = image.resize(new_size, Image.Resampling.LANCZOS)
    
    text = pytesseract.image_to_string(image)
    image.close()  # Free memory
    
    return text
```

## API Reference

### Main Functions
```python
# Core OCR functions
pytesseract.image_to_string(image, lang='eng', config='', nice=0, output_type=Output.STRING, timeout=0)
pytesseract.image_to_data(image, lang='eng', config='', nice=0, output_type=Output.STRING, timeout=0)
pytesseract.image_to_boxes(image, lang='eng', config='', nice=0, output_type=Output.STRING, timeout=0)
pytesseract.image_to_osd(image, config='', nice=0, output_type=Output.STRING, timeout=0)

# Utility functions
pytesseract.get_languages(config='')
pytesseract.get_tesseract_version()
```

### Output Types
```python
pytesseract.Output.BYTES
pytesseract.Output.DATAFRAME  # Requires pandas
pytesseract.Output.DICT
pytesseract.Output.STRING
```

### Configuration Parameters
- `lang`: Language code(s) (e.g., 'eng', 'eng+fra')
- `config`: Tesseract configuration string
- `nice`: Process priority (Unix only)
- `output_type`: Format of returned data
- `timeout`: Maximum execution time in seconds

### Common Config Options
- `--psm N`: Page segmentation mode (0-13)
- `--oem N`: OCR Engine mode (0-3)
- `-c CONFIGVAR=VALUE`: Set configuration variable
- `--tessdata-dir PATH`: Specify tessdata directory
- `-l LANG`: Language (alternative to lang parameter)

This comprehensive documentation covers all aspects of using pytesseract for OCR tasks in Python applications.
