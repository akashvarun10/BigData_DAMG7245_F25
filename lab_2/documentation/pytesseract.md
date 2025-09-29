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
