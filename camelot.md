# Camelot-py Documentation

Camelot is a Python library that makes it easy to extract tables from PDF files. It provides both lattice-based and stream-based parsing methods to handle different types of table layouts in PDFs.

## Table of Contents
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Basic Usage](#basic-usage)
- [Parsing Methods](#parsing-methods)
- [Advanced Configuration](#advanced-configuration)
- [Table Processing](#table-processing)
- [Export Options](#export-options)
- [Visual Debugging](#visual-debugging)
- [Performance Tips](#performance-tips)
- [Common Issues](#common-issues)
- [API Reference](#api-reference)

## Installation

### Basic Installation
```bash
pip install camelot-py[base]
```

### Full Installation (Recommended)
```bash
pip install camelot-py[cv]
```

### System Dependencies

#### Ubuntu/Debian
```bash
sudo apt install python3-tk ghostscript
```

#### macOS
```bash
brew install ghostscript tcl-tk
```

#### Windows
- Install [Ghostscript](https://www.ghostscript.com/download/gsdnld.html)
- Ensure it's added to your PATH

### Verify Installation
```python
import camelot
print(camelot.__version__)
```

## Quick Start

### Basic Table Extraction
```python
import camelot

# Extract tables from PDF
tables = camelot.read_pdf('example.pdf')

# Print number of tables found
print(f"Total tables extracted: {tables.n}")

# Access first table
table = tables[0]

# Convert to pandas DataFrame
df = table.df
print(df.head())
```

### Save to File
```python
# Export to CSV
tables.export('output.csv', f='csv')

# Export to Excel
tables.export('output.xlsx', f='excel')

# Export to JSON
tables.export('output.json', f='json')
```

## Basic Usage

### Reading PDF Files
```python
import camelot

# Read all pages
tables = camelot.read_pdf('file.pdf')

# Read specific pages
tables = camelot.read_pdf('file.pdf', pages='1,2,3')

# Read page range
tables = camelot.read_pdf('file.pdf', pages='1-5')

# Read all pages except specific ones
tables = camelot.read_pdf('file.pdf', pages='all', 
                         exclude_pages='1,2')
```

### Accessing Table Data
```python
# Get number of tables
num_tables = tables.n

# Iterate through tables
for i, table in enumerate(tables):
    print(f"Table {i+1}:")
    print(f"Shape: {table.shape}")
    print(f"Accuracy: {table.accuracy}")
    print(f"Whitespace: {table.whitespace}")
    print(table.df.head())
    print("-" * 50)

# Access specific table
first_table = tables[0]
last_table = tables[-1]

# Get table as DataFrame
df = tables[0].df

# Get table as list of lists
data = tables[0].data
```

## Parsing Methods

### Lattice Method (Default)
Best for tables with clear borders and grid lines.

```python
# Default lattice parsing
tables = camelot.read_pdf('file.pdf', flavor='lattice')

# With custom parameters
tables = camelot.read_pdf('file.pdf', 
                         flavor='lattice',
                         table_areas=['10,85,563,15'],  # x1,y1,x2,y2
                         columns=['10,85,563,15'])      # column separators
```

#### Lattice Parameters
```python
tables = camelot.read_pdf('file.pdf',
                         flavor='lattice',
                         table_areas=['10,85,563,15'],
                         columns=['72,95,209,327,442,538'],
                         split_text=True,
                         flag_size=True,
                         strip_text=' .\n',
                         row_tol=2,
                         col_tol=0)
```

### Stream Method
Best for tables without clear borders, using whitespace separation.

```python
# Stream parsing
tables = camelot.read_pdf('file.pdf', flavor='stream')

# With custom parameters
tables = camelot.read_pdf('file.pdf',
                         flavor='stream',
                         table_areas=['10,85,563,15'],
                         columns=['72,95,209,327,442,538'],
                         row_tol=2,
                         col_tol=0)
```

#### Stream Parameters
```python
tables = camelot.read_pdf('file.pdf',
                         flavor='stream',
                         table_areas=['316,499,566,337'],
                         columns=['72,95,209,327,442,538'],
                         edge_tol=500,
                         row_tol=2,
                         col_tol=0,
                         split_text=True,
                         flag_size=True,
                         strip_text=' .\n')
```

## Advanced Configuration

### Table Areas
Specify exact regions to extract tables from:

```python
# Single table area
tables = camelot.read_pdf('file.pdf',
                         table_areas=['72,95,209,327'])

# Multiple table areas
tables = camelot.read_pdf('file.pdf',
                         table_areas=['72,95,209,327',
                                    '442,538,563,445'])

# Different areas for different pages
tables = camelot.read_pdf('file.pdf',
                         pages='1-3',
                         table_areas=['72,95,209,327',  # page 1
                                    '10,85,563,15',   # page 2
                                    '20,75,553,25'])  # page 3
```

### Column Specification
Define column boundaries manually:

```python
# Specify column separators
tables = camelot.read_pdf('file.pdf',
                         columns=['72,95,209,327,442,538'])

# Different columns for different pages
tables = camelot.read_pdf('file.pdf',
                         pages='1,2',
                         columns=['72,95,209,327',     # page 1
                                '10,85,563,15'])      # page 2
```

### Text Processing Options
```python
tables = camelot.read_pdf('file.pdf',
                         split_text=True,        # Split text by newlines
                         flag_size=True,         # Flag text size changes
                         strip_text=' .\n',      # Characters to strip
                         row_tol=2,             # Row tolerance
                         col_tol=0)             # Column tolerance
```

### Password-Protected PDFs
```python
tables = camelot.read_pdf('protected.pdf', password='secret123')
```

## Table Processing

### Table Properties
```python
table = tables[0]

# Basic properties
print(f"Shape: {table.shape}")           # (rows, columns)
print(f"Accuracy: {table.accuracy}")     # Parsing accuracy score
print(f"Whitespace: {table.whitespace}") # Whitespace ratio
print(f"Order: {table.order}")           # Table order on page
print(f"Page: {table.page}")             # Page number

# Bounding box
print(f"Bbox: {table.bbox}")             # (x1, y1, x2, y2)
```

### Data Manipulation
```python
# Get DataFrame
df = table.df

# Clean data
df = df.replace('', None)  # Replace empty strings with None
df = df.dropna(how='all')  # Drop empty rows
df = df.dropna(axis=1, how='all')  # Drop empty columns

# Set column headers
if not df.empty:
    df.columns = df.iloc[0]  # Use first row as headers
    df = df.drop(df.index[0])  # Remove header row from data

# Reset index
df = df.reset_index(drop=True)
```

### Filtering Tables
```python
# Filter by accuracy
high_accuracy_tables = [t for t in tables if t.accuracy > 80]

# Filter by size
large_tables = [t for t in tables if t.shape[0] > 5 and t.shape[1] > 3]

# Filter by whitespace ratio
dense_tables = [t for t in tables if t.whitespace < 50]
```

## Export Options

### Single Table Export
```python
table = tables[0]

# Export to different formats
table.to_csv('table.csv')
table.to_excel('table.xlsx')
table.to_json('table.json')
table.to_html('table.html')
```

### Multiple Tables Export
```python
# Export all tables
tables.export('output.csv', f='csv')
tables.export('output.xlsx', f='excel') 
tables.export('output.json', f='json')
tables.export('output.html', f='html')

# Export with compression
tables.export('output.csv', f='csv', compress=True)

# Export specific tables
selected_tables = camelot.TableList([tables[0], tables[2]])
selected_tables.export('selected.csv', f='csv')
```

### Custom Export Parameters
```python
# CSV with custom separator
tables.export('output.csv', f='csv', sep='|')

# Excel with custom sheet names
tables.export('output.xlsx', f='excel', 
              sheet_name=['Table1', 'Table2', 'Table3'])

# JSON with custom formatting
tables.export('output.json', f='json', orient='records')
```

## Visual Debugging

### Plot Tables
```python
import matplotlib.pyplot as plt

# Plot table boundaries
camelot.plot(tables[0], kind='contour')
plt.show()

# Plot detected grid lines
camelot.plot(tables[0], kind='grid')
plt.show()

# Plot text boundaries
camelot.plot(tables[0], kind='text')
plt.show()
```

### Save Plots
```python
# Save plot to file
camelot.plot(tables[0], kind='contour').savefig('table_plot.png')

# Multiple plots
for i, table in enumerate(tables):
    plot = camelot.plot(table, kind='contour')
    plot.savefig(f'table_{i+1}_plot.png')
    plt.close()
```

### Generate Report
```python
# Generate parsing report
tables[0].parsing_report

# Detailed report with plots
report = camelot.plot(tables[0], kind='contour')
report.savefig('parsing_report.png', bbox_inches='tight', dpi=300)
```

## Performance Tips

### Memory Management
```python
# Process large PDFs page by page
def process_large_pdf(filename):
    all_tables = []
    
    # Get total pages (you might need PyPDF2 for this)
    for page_num in range(1, total_pages + 1):
        tables = camelot.read_pdf(filename, pages=str(page_num))
        
        # Process tables immediately
        for table in tables:
            if table.accuracy > 80:  # Quality filter
                processed_df = clean_table(table.df)
                all_tables.append(processed_df)
        
        # Clear memory
        del tables
    
    return all_tables
```

### Optimization Settings
```python
# For better performance with large files
tables = camelot.read_pdf('large_file.pdf',
                         pages='1-10',          # Process in chunks
                         flavor='lattice',      # Usually faster
                         copy_text=['v'],       # Reduce text processing
                         table_areas=['auto'],  # Let camelot detect
                         edge_tol=500)          # Adjust tolerance
```

### Parallel Processing
```python
from concurrent.futures import ProcessPoolExecutor
import os

def extract_tables_from_page(args):
    filename, page = args
    return camelot.read_pdf(filename, pages=str(page))

def parallel_extraction(filename, pages):
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        args = [(filename, page) for page in pages]
        results = list(executor.map(extract_tables_from_page, args))
    
    # Combine results
    all_tables = camelot.TableList()
    for result in results:
        all_tables.extend(result)
    
    return all_tables
```

## Common Issues

### No Tables Found
```python
# Check if tables were found
if tables.n == 0:
    print("No tables found. Try:")
    print("1. Different flavor (lattice/stream)")
    print("2. Specify table_areas manually")
    print("3. Check if PDF has selectable text")
    
    # Try alternative approach
    tables = camelot.read_pdf('file.pdf', flavor='stream')
```

### Poor Accuracy
```python
# Improve accuracy
tables = camelot.read_pdf('file.pdf',
                         flavor='lattice',
                         table_areas=['manual_coordinates'],
                         columns=['column_coordinates'],
                         split_text=True,
                         flag_size=True)

# Check accuracy
for i, table in enumerate(tables):
    if table.accuracy < 80:
        print(f"Table {i+1} has low accuracy: {table.accuracy}%")
        # Consider manual area specification
```

### Merged Cells Issues
```python
# Handle merged cells
def fix_merged_cells(df):
    # Forward fill merged cells
    df = df.fillna(method='ffill', axis=0)  # Fill down
    df = df.fillna(method='ffill', axis=1)  # Fill right
    return df

# Apply to all tables
for table in tables:
    df = fix_merged_cells(table.df)
```

### Text Encoding Issues
```python
# Handle encoding issues
def clean_text(df):
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].astype(str).str.encode('utf-8', errors='ignore').str.decode('utf-8')
    return df
```

## API Reference

### Main Functions
```python
# Primary function
camelot.read_pdf(filepath, pages='1', password=None, flavor='lattice', **kwargs)

# Plotting function  
camelot.plot(table, kind='contour', **kwargs)
```

### TableList Methods
```python
tables = camelot.read_pdf('file.pdf')

# Properties
tables.n                    # Number of tables
tables[index]              # Access table by index
len(tables)                # Number of tables

# Methods
tables.export(filename, f='csv', **kwargs)  # Export all tables
tables.extend(other_tables)                 # Combine table lists
```

### Table Methods
```python
table = tables[0]

# Properties
table.df                   # Pandas DataFrame
table.data                 # List of lists
table.shape                # (rows, columns)
table.accuracy             # Parsing accuracy (0-100)
table.whitespace           # Whitespace ratio (0-100)
table.order                # Table order on page
table.page                 # Page number
table.bbox                 # Bounding box coordinates

# Methods
table.to_csv(filename, **kwargs)      # Export to CSV
table.to_excel(filename, **kwargs)    # Export to Excel
table.to_json(filename, **kwargs)     # Export to JSON
table.to_html(filename, **kwargs)     # Export to HTML
```

### Parameters Reference

#### Common Parameters
- `pages`: str - Page numbers ('1', '1,2,3', '1-5', 'all')
- `password`: str - PDF password
- `flavor`: str - Parsing method ('lattice' or 'stream')
- `table_areas`: list - Table regions as ['x1,y1,x2,y2']
- `columns`: list - Column separators as ['x1,x2,x3']
- `split_text`: bool - Split text by newlines
- `flag_size`: bool - Flag text size changes
- `strip_text`: str - Characters to strip from text
- `row_tol`: int - Row tolerance for grouping
- `col_tol`: int - Column tolerance for grouping

#### Lattice-Specific Parameters
- `process_background`: bool - Process background lines
- `line_scale`: int - Line detection scaling factor
- `copy_text`: list - Text copying method
- `shift_text`: list - Text shifting method

#### Stream-Specific Parameters  
- `edge_tol`: int - Edge tolerance for text grouping
- `layout_kwargs`: dict - Additional layout parameters

This documentation provides comprehensive coverage of Camelot-py for PDF table extraction workflows.
