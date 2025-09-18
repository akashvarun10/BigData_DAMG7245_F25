#!/usr/bin/env python3
"""
Docling PDF Processor

A modular tool to process PDF documents and extract:
- Markdown content
- Images (figures and table images) to images/ folder  
- Tables (CSV and HTML) to tables/ folder

Usage:
    python docling_processor.py input.pdf output_directory
"""

import logging
import time
from pathlib import Path
from typing import Optional
import pandas as pd

from docling_core.types.doc import ImageRefMode, PictureItem, TableItem
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption


def setup_logging(level: int = logging.INFO) -> logging.Logger:
    """Setup logging configuration."""
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)


def create_document_converter(image_scale: float = 2.0) -> DocumentConverter:
    """
    Create and configure a DocumentConverter for PDF processing.
    
    Args:
        image_scale: Scale factor for images (1.0 = 72 DPI, 2.0 = 144 DPI)
        
    Returns:
        Configured DocumentConverter instance
    """
    pipeline_options = PdfPipelineOptions()
    pipeline_options.images_scale = image_scale
    pipeline_options.generate_page_images = True
    pipeline_options.generate_picture_images = True
    
    return DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )


def export_markdown(conversion_result, output_dir: Path, filename_stem: str) -> Path:
    """
    Export document to markdown format.
    
    Args:
        conversion_result: Docling conversion result
        output_dir: Output directory
        filename_stem: Base filename without extension
        
    Returns:
        Path to the saved markdown file
    """
    md_path = output_dir / f"{filename_stem}.md"
    
    # Export markdown with embedded images
    conversion_result.document.save_as_markdown(
        md_path, 
        image_mode=ImageRefMode.REFERENCED
    )
    
    return md_path


def export_images(conversion_result, output_dir: Path, filename_stem: str, logger: logging.Logger) -> dict:
    """
    Export images (figures and table images) to images folder.
    
    Args:
        conversion_result: Docling conversion result
        output_dir: Output directory
        filename_stem: Base filename without extension
        logger: Logger instance
        
    Returns:
        Dictionary with export statistics
    """
    images_dir = output_dir / "images"
    images_dir.mkdir(exist_ok=True)
    
    table_counter = 0
    picture_counter = 0
    
    # Extract individual elements (figures and tables as images)
    for element, _ in conversion_result.document.iterate_items():
        if isinstance(element, TableItem):
            table_counter += 1
            image_path = images_dir / f"{filename_stem}-table-{table_counter}.png"
            try:
                with image_path.open("wb") as fp:
                    element.get_image(conversion_result.document).save(fp, "PNG")
                logger.info(f"Saved table image: {image_path}")
            except Exception as e:
                logger.warning(f"Could not save table image {table_counter}: {e}")
                
        elif isinstance(element, PictureItem):
            picture_counter += 1
            image_path = images_dir / f"{filename_stem}-picture-{picture_counter}.png"
            try:
                with image_path.open("wb") as fp:
                    element.get_image(conversion_result.document).save(fp, "PNG")
                logger.info(f"Saved picture image: {image_path}")
            except Exception as e:
                logger.warning(f"Could not save picture image {picture_counter}: {e}")
    
    return {
        "tables_as_images": table_counter,
        "pictures": picture_counter,
        "images_dir": images_dir
    }


def export_tables(conversion_result, output_dir: Path, filename_stem: str, logger: logging.Logger) -> dict:
    """
    Export tables to tables folder as CSV and HTML.
    
    Args:
        conversion_result: Docling conversion result
        output_dir: Output directory
        filename_stem: Base filename without extension
        logger: Logger instance
        
    Returns:
        Dictionary with export statistics
    """
    tables_dir = output_dir / "tables"
    tables_dir.mkdir(exist_ok=True)
    
    table_count = 0
    
    for table_ix, table in enumerate(conversion_result.document.tables):
        table_count += 1
        
        try:
            # Convert to pandas DataFrame
            table_df = table.export_to_dataframe()
            
            # Save as CSV
            csv_path = tables_dir / f"{filename_stem}-table-{table_ix + 1}.csv"
            table_df.to_csv(csv_path, index=False)
            logger.info(f"Saved table CSV: {csv_path}")
            
            # Save as HTML
            html_path = tables_dir / f"{filename_stem}-table-{table_ix + 1}.html"
            with html_path.open("w", encoding="utf-8") as fp:
                fp.write(table.export_to_html(doc=conversion_result.document))
            logger.info(f"Saved table HTML: {html_path}")
            
        except Exception as e:
            logger.warning(f"Could not export table {table_ix + 1}: {e}")
    
    return {
        "table_count": table_count,
        "tables_dir": tables_dir
    }


def process_pdf(
    pdf_path: str | Path, 
    output_dir: str | Path, 
    image_scale: float = 2.0
) -> dict:
    """
    Main function to process PDF and export all content.
    
    Args:
        pdf_path: Path to input PDF file
        output_dir: Output directory path
        image_scale: Scale factor for images (default: 2.0 for 144 DPI)
        
    Returns:
        Dictionary with processing results and statistics
    """
    logger = setup_logging()
    
    # Convert paths
    pdf_path = Path(pdf_path)
    output_dir = Path(output_dir)
    
    # Validate input
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    
    if not pdf_path.suffix.lower() == '.pdf':
        raise ValueError(f"Input file must be a PDF: {pdf_path}")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    filename_stem = pdf_path.stem
    
    logger.info(f"Processing PDF: {pdf_path}")
    logger.info(f"Output directory: {output_dir}")
    
    # Initialize converter and process document
    start_time = time.time()
    converter = create_document_converter(image_scale)
    
    try:
        conversion_result = converter.convert(str(pdf_path))
    except Exception as e:
        logger.error(f"Failed to convert PDF: {e}")
        raise
    
    # Export markdown
    try:
        md_path = export_markdown(conversion_result, output_dir, filename_stem)
        logger.info(f"Exported markdown: {md_path}")
    except Exception as e:
        logger.error(f"Failed to export markdown: {e}")
        md_path = None
    
    # Export images
    try:
        image_stats = export_images(conversion_result, output_dir, filename_stem, logger)
        logger.info(f"Exported {image_stats['pictures']} pictures and {image_stats['tables_as_images']} table images")
    except Exception as e:
        logger.error(f"Failed to export images: {e}")
        image_stats = {"error": str(e)}
    
    # Export tables
    try:
        table_stats = export_tables(conversion_result, output_dir, filename_stem, logger)
        logger.info(f"Exported {table_stats['table_count']} tables")
    except Exception as e:
        logger.error(f"Failed to export tables: {e}")
        table_stats = {"error": str(e)}
    
    processing_time = time.time() - start_time
    logger.info(f"Processing completed in {processing_time:.2f} seconds")
    
    return {
        "pdf_path": str(pdf_path),
        "output_dir": str(output_dir),
        "processing_time": processing_time,
        "markdown_path": str(md_path) if md_path else None,
        "image_stats": image_stats,
        "table_stats": table_stats
    }


def main():
    """Command line interface."""
    import sys
    
    if len(sys.argv) != 3:
        print("Usage: python docling_processor.py <pdf_path> <output_directory>")
        print("Example: python docling_processor.py document.pdf ./output")
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    output_dir = sys.argv[2]
    
    try:
        result = process_pdf(pdf_path, output_dir)
        print("\n" + "="*50)
        print("PROCESSING COMPLETE")
        print("="*50)
        print(f"PDF: {result['pdf_path']}")
        print(f"Output: {result['output_dir']}")
        print(f"Time: {result['processing_time']:.2f}s")
        
        if result['markdown_path']:
            print(f"Markdown: {result['markdown_path']}")
        
        if 'table_count' in result['table_stats']:
            print(f"Tables exported: {result['table_stats']['table_count']}")
        
        if 'pictures' in result['image_stats']:
            print(f"Images exported: {result['image_stats']['pictures'] + result['image_stats']['tables_as_images']}")
            
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()