# Typer CLI - Beginner's Guide

## What is Typer?

Typer is a modern Python library for building command-line interfaces (CLIs) that are easy to create and delightful to use. It leverages Python's type hints to automatically generate help text, validate inputs, and provide auto-completion.

## Installation

### Basic Installation
```bash
pip install typer
```

This installs Typer with all recommended dependencies including:
- **Rich**: For beautiful formatted output and error messages
- **Shellingham**: For automatic shell detection and completion

### Verify Installation
```bash
typer --version
```

## Three Ways to Use Typer

### 1. 🎯 The Simplest Way (No Import Needed)

**File: `hello_basic.py`**
```python
# hello_basic.py - doesn't even need to import typer
def main(name: str):
    print(f"Hello {name}")
```

**How to Run:**
```bash
# Run the script using the typer command
typer hello_basic.py run Camila
# Output: Hello Camila

# Get help
typer hello_basic.py run --help
```

**What's Happening:**
- No need to import typer in your script
- The `typer` command automatically converts your function to a CLI
- Type hints (`name: str`) tell Typer what kind of input to expect

### 2. 🚀 Using `typer.run()` (Single Command)

**File: `main.py`**
```python
import typer

def main(name: str):
    print(f"Hello {name}")

if __name__ == "__main__":
    typer.run(main)
```

**How to Run:**
```bash
# Run directly with Python
python main.py Camila
# Output: Hello Camila

# Get help
python main.py --help
```

**What's Happening:**
- `typer.run(main)` converts your single function into a CLI app
- More control than method 1, but still simple
- Works with standard `python` command

### 3. 🏗️ Multi-Command Application (Full Power)

**File: `app.py`**
```python
import typer

app = typer.Typer()

@app.command()
def hello(name: str):
    print(f"Hello {name}")

@app.command()
def goodbye(name: str, formal: bool = False):
    if formal:
        print(f"Goodbye Ms. {name}. Have a good day.")
    else:
        print(f"Bye {name}!")

if __name__ == "__main__":
    app()
```

**How to Run:**
```bash
# See all available commands
python app.py --help

# Run the hello command
python app.py hello Camila
# Output: Hello Camila

# Run goodbye command (casual)
python app.py goodbye Alice
# Output: Bye Alice!

# Run goodbye command (formal)
python app.py goodbye Sarah --formal
# Output: Goodbye Ms. Sarah. Have a good day.

# Get help for specific command
python app.py goodbye --help
```

## Understanding Arguments vs Options

### 🔑 Arguments (Required Parameters)

Arguments are **required** values that your command needs to work.

```python
def hello(name: str):  # 'name' is an ARGUMENT
    print(f"Hello {name}")
```

**Usage:**
```bash
python app.py hello "John"  # "John" is the argument
#                    ^^^^^^ This is required!
```

**Characteristics:**
- Always required (unless you give them default values)
- Positional (order matters)
- No `--` prefix needed
- Show up as `Arguments:` in help text

### ⚙️ Options (Optional Parameters)

Options are **optional** values that modify how your command behaves.

```python
def goodbye(name: str, formal: bool = False):
    #                   ^^^^^^^^^^^^^^^^^^^ This is an OPTION
    print("Goodbye message")
```

**Usage:**
```bash
python app.py goodbye "Alice"           # formal=False (default)
python app.py goodbye "Alice" --formal  # formal=True
python app.py goodbye "Alice" --no-formal  # formal=False (explicit)
```

**Characteristics:**
- Optional (have default values)
- Use `--` prefix
- Can appear in any order after arguments
- Show up as `Options:` in help text

## Common Parameter Types

### String Arguments
```python
def greet(name: str):
    print(f"Hello {name}")

# Usage: python app.py greet "Alice"
```

### Integer Arguments
```python
def repeat(message: str, times: int):
    for _ in range(times):
        print(message)

# Usage: python app.py repeat "Hi" 3
```

### Boolean Options (Flags)
```python
def process(data: str, verbose: bool = False):
    if verbose:
        print("Processing in verbose mode")
    print(f"Processing {data}")

# Usage: python app.py process "data.txt" --verbose
```

### Optional String Parameters
```python
def greet(name: str, greeting: str = "Hello"):
    print(f"{greeting} {name}")

# Usage: 
# python app.py greet "Alice"                    # Uses default "Hello"
# python app.py greet "Alice" --greeting "Hi"   # Custom greeting
```

## Help System

Typer automatically generates help text for your applications!

### Automatic Help
```bash
# See all commands
python app.py --help

# See help for specific command
python app.py hello --help
```

### Custom Help with Docstrings
```python
@app.command()
def hello(name: str):
    """Say hello to someone."""
    print(f"Hello {name}")

@app.command()  
def goodbye(name: str, formal: bool = False):
    """
    Say goodbye to someone.
    
    Use --formal for a more professional goodbye.
    """
    if formal:
        print(f"Goodbye Ms. {name}. Have a good day.")
    else:
        print(f"Bye {name}!")
```

## Quick Reference Commands

### Getting Help
```bash
# Application help
python app.py --help

# Command-specific help  
python app.py [COMMAND] --help

# Example
python app.py goodbye --help
```

### Running Commands
```bash
# Single command app
python single_app.py [ARGUMENTS] [OPTIONS]

# Multi-command app
python multi_app.py [COMMAND] [ARGUMENTS] [OPTIONS]

# Examples
python app.py hello "Alice"
python app.py goodbye "Bob" --formal
```

### Auto-completion Setup
```bash
# Install completion for your shell
python app.py --install-completion

# Show completion script (for manual setup)
python app.py --show-completion
```

## Common Patterns for Beginners

### 1. Simple Script with One Required Argument
```python
import typer

def main(filename: str):
    """Process a file."""
    print(f"Processing file: {filename}")

if __name__ == "__main__":
    typer.run(main)
```

### 2. Script with Optional Configuration
```python
import typer

def main(input_file: str, output_file: str = "output.txt", verbose: bool = False):
    """Convert input file to output file."""
    if verbose:
        print(f"Converting {input_file} to {output_file}")
    
    # Your processing logic here
    print("Done!")

if __name__ == "__main__":
    typer.run(main)
```

### 3. Multi-Command Tool
```python
import typer

app = typer.Typer()

@app.command()
def create(name: str):
    """Create a new item."""
    print(f"Creating: {name}")

@app.command()
def delete(name: str, force: bool = False):
    """Delete an item."""
    if force or typer.confirm(f"Delete {name}?"):
        print(f"Deleted: {name}")
    else:
        print("Cancelled")

if __name__ == "__main__":
    app()
```

## Troubleshooting

### Common Errors

**1. "Missing argument"**
```bash
# ❌ Error: Missing required argument
python app.py hello
# ✅ Fix: Provide the required argument
python app.py hello "Alice"
```

**2. "No such option"**
```bash
# ❌ Error: Option in wrong place
python app.py hello --formal "Alice"
# ✅ Fix: Arguments come before options
python app.py hello "Alice" --formal
```

**3. "No such command"**
```bash
# ❌ Error: Wrong command name
python app.py greet "Alice"
# ✅ Fix: Use correct command name
python app.py hello "Alice"
```

## Next Steps

Once you're comfortable with the basics:
- Explore advanced features like custom validation
- Learn about progress bars and rich formatting
- Add configuration files to your apps
- Create installable packages with entry points
- Use Typer with FastAPI for web + CLI apps

## Resources

- 📚 [Official Typer Documentation](https://typer.tiangolo.com/)
- 🐍 [Python Type Hints Guide](https://docs.python.org/3/library/typing.html)
- 💡 [More Examples and Tutorials](https://typer.tiangolo.com/tutorial/)

---

