# Docling PDF Processor

A clean, modular Python tool for processing PDF documents using IBM's Docling library. Extracts and organizes content into markdown, images, and structured tables.

## Features

- **Markdown Export**: Clean markdown with preserved structure
- **Image Extraction**: Figures and table images saved to `images/` folder
- **Table Export**: Tables saved as CSV and HTML files in `tables/` folder
- **Modular Design**: Clean, reusable functions with proper error handling
- **Batch Processing**: Process multiple PDFs efficiently

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Or install directly
pip install docling pandas
```

## Usage

### Command Line Interface

```bash
# Basic usage
python docling_processor.py input.pdf output_directory

# Example
python docling_processor.py research_paper.pdf ./extracted_content
```

### Python API

```python
from docling_processor import process_pdf

# Process single PDF
result = process_pdf("document.pdf", "./output")

# Custom image resolution (higher DPI)
result = process_pdf(
    pdf_path="document.pdf",
    output_dir="./output",
    image_scale=3.0  # 216 DPI
)
```

## Output Structure

```
output_directory/
├── document_name.md              # Markdown export
├── images/                       # Images folder
│   ├── document_name-picture-1.png
│   ├── document_name-table-1.png
│   └── ...
└── tables/                       # Tables folder
    ├── document_name-table-1.csv
    ├── document_name-table-1.html
    └── ...
```

## Key Functions

### `process_pdf(pdf_path, output_dir, image_scale=2.0)`
Main processing function that orchestrates the entire workflow.

### `export_markdown(conversion_result, output_dir, filename_stem)`
Exports document content to markdown format with referenced images.

### `export_images(conversion_result, output_dir, filename_stem, logger)`
Extracts and saves all images (figures and table images) as PNG files.

### `export_tables(conversion_result, output_dir, filename_stem, logger)`
Exports tables as both CSV and HTML files with proper formatting.

## Configuration

- **Image Scale**: Control output image resolution (1.0 = 72 DPI, 2.0 = 144 DPI, 3.0 = 216 DPI)
- **Logging**: Built-in logging shows processing progress and any warnings
- **Error Handling**: Graceful handling of missing images or malformed tables

## Requirements

- Python 3.8+
- docling>=2.53.0
- pandas>=1.5.0
- Optional: OCR support via `docling[easyocr]` or `docling[paddle]`

## Performance Notes

- First run downloads AI models (~2.5 minutes initial setup)
- Subsequent runs use cached models
- Processing is computationally intensive; GPU recommended for large documents
- Higher image scales increase processing time but improve quality

## Examples

See `docling_usage_example.py` for detailed usage examples including:
- Basic single PDF processing
- Batch processing multiple PDFs
- Custom resolution settings
- Error handling patterns

## License

This tool uses the open-source Docling library by IBM Research (MIT License).