# Docling PDF Processor

## What is Docling?

Docling is an open-source document processing toolkit developed by IBM Research that simplifies document processing, parsing diverse formats — including advanced PDF understanding — and providing seamless integrations with the gen AI ecosystem. It's designed to prepare documents for generative AI applications like RAG (Retrieval-Augmented Generation).
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

I'll help you understand Docling and its figure and table export capabilities. Let me search for information about this tool.## Understanding Docling: Document Processing for AI

Based on my research, here's a comprehensive overview of Docling and its figure and table export capabilities:



### Key Features:
- **Multi-format support**: PDF, DOCX, PPTX, XLSX, HTML, WAV, MP3, images (PNG, TIFF, JPEG, etc.)
- **Advanced PDF understanding**: Page layout, reading order, table structure, code, formulas, image classification
- **Export formats**: Markdown, HTML, DocTags, and lossless JSON
- **AI model integrations**: LangChain, LlamaIndex, Crew AI, Haystack
- **Local execution**: Supports air-gapped environments for sensitive data

## Core AI Models

Docling uses a set of pre-trained models hosted on Hugging Face for layout recognition, table parsing, image classification, and OCR:

- **Layout Analysis Model**: Object detector for page elements (DocLayNet)
- **Table Structure Parsing**: TableFormer model for reconstructing table structures
- **Image Classification**: Fine-tuned EfficientNet-B0 for 16 categories (charts, maps, logos, etc.)
- **Code & Formula Extraction**: Processes code snippets and mathematical formulas
- **Visual Language Model**: SmolVLM for image analysis and description

## Figure Export Capabilities

### Basic Setup for Figure Export:

```python
from docling_core.types.doc import ImageRefMode, PictureItem, TableItem
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption

# Configure pipeline for image export
IMAGE_RESOLUTION_SCALE = 2.0
pipeline_options = PdfPipelineOptions()
pipeline_options.images_scale = IMAGE_RESOLUTION_SCALE
pipeline_options.generate_page_images = True
pipeline_options.generate_picture_images = True

doc_converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
    }
)
```

### Figure Export Methods:

**1. Extract Individual Figures:**
You can save images of figures and tables by iterating through document elements and extracting them as PNG files:

```python
# Save images of figures and tables
table_counter = 0
picture_counter = 0
for element, _level in conv_res.document.iterate_items():
    if isinstance(element, TableItem):
        table_counter += 1
        element_image_filename = output_dir / f"{doc_filename}-table-{table_counter}.png"
        with element_image_filename.open("wb") as fp:
            element.get_image(conv_res.document).save(fp, "PNG")
    
    if isinstance(element, PictureItem):
        picture_counter += 1
        element_image_filename = output_dir / f"{doc_filename}-picture-{picture_counter}.png"
        with element_image_filename.open("wb") as fp:
            element.get_image(conv_res.document).save(fp, "PNG")
```

**2. Export with Different Image Reference Modes:**
Docling supports multiple ways to handle images in exported documents:

```python
# Save markdown with embedded pictures
conv_res.document.save_as_markdown(md_filename, image_mode=ImageRefMode.EMBEDDED)

# Save markdown with externally referenced pictures
conv_res.document.save_as_markdown(md_filename, image_mode=ImageRefMode.REFERENCED)

# Save HTML with externally referenced pictures
conv_res.document.save_as_html(html_filename, image_mode=ImageRefMode.REFERENCED)
```

## Table Export Capabilities

### Basic Table Export:

Docling provides comprehensive table extraction and export functionality:

```python
from docling.document_converter import DocumentConverter
import pandas as pd

doc_converter = DocumentConverter()
conv_res = doc_converter.convert(input_doc_path)

# Export tables
for table_ix, table in enumerate(conv_res.document.tables):
    # Convert to pandas DataFrame
    table_df: pd.DataFrame = table.export_to_dataframe()
    
    # Print as Markdown
    print(f"## Table {table_ix}")
    print(table_df.to_markdown())
    
    # Save as CSV
    table_df.to_csv(f"{doc_filename}-table-{table_ix + 1}.csv")
    
    # Save as HTML
    with open(f"{doc_filename}-table-{table_ix + 1}.html", "w") as fp:
        fp.write(table.export_to_html(doc=conv_res.document))
```

### Table Export Formats:
- **Pandas DataFrame**: Direct integration with data science workflows
- **CSV**: Standard comma-separated values
- **HTML**: Rich formatting preserved
- **Markdown**: Table format for documentation




**Export Modes:**
- `ExportType.DOC_CHUNKS`: Each input document chunked into separate LangChain Documents
- `ExportType.MARKDOWN`: Each input document as a single LangChain Document

## Performance and Hardware Requirements

Docling can be computationally intensive, especially for complex documents. Having GPU access is helpful, though not required. It can avoid OCR when possible, which reduces errors and speeds up processing by up to 30 times compared to traditional OCR approaches.

**Key Benefits:**
- **Speed**: 30x faster than traditional OCR methods when OCR can be avoided
- **Accuracy**: Advanced AI models provide better layout and structure recognition
- **Flexibility**: Multiple export formats and integration options
- **Enterprise-ready**: Local execution for sensitive data

Docling represents a significant advancement in document processing for AI applications, offering robust figure and table extraction capabilities that make it particularly valuable for preparing enterprise documents for generative AI workflows.