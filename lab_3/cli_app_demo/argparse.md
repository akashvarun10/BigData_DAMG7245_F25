# Simple Argparse CLI Documentation

## What This Code Does

This is a simple command-line program with two commands: `hello` and `goodbye`. It uses Python's built-in `argparse` module to handle user input.

## Code Explanation

```python
import argparse

def main():
    # Create the main parser
    parser = argparse.ArgumentParser(description="Simple CLI with argparse")
    
    # Add subcommands (hello and goodbye)
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Setup "hello" command
    hello_parser = subparsers.add_parser('hello', help='Say hello')
    hello_parser.add_argument('name', help='Name to greet')  # Required argument
    
    # Setup "goodbye" command  
    goodbye_parser = subparsers.add_parser('goodbye', help='Say goodbye')
    goodbye_parser.add_argument('name', help='Name to say goodbye to')  # Required
    goodbye_parser.add_argument('--formal', action='store_true', help='Formal goodbye')  # Optional flag
    
    # Parse what the user typed
    args = parser.parse_args()
    
    # Do different things based on the command
    if args.command == 'hello':
        print(f"Hello {args.name}")
    
    elif args.command == 'goodbye':
        if args.formal:  # If --formal flag was used
            print(f"Goodbye Ms. {args.name}. Have a good day.")
        else:
            print(f"Bye {args.name}!")
    
    else:
        parser.print_help()  # Show help if no command given

if __name__ == "__main__":
    main()
```

## How to Run

Save the code as `simple_argparse.py` and run these commands:

### 1. Get Help

**Command:**
```bash
python simple_argparse.py --help
```

**Output:**
```
usage: simple_argparse.py [-h] {hello,goodbye} ...

Simple CLI with argparse

positional arguments:
  {hello,goodbye}  Available commands
    hello          Say hello
    goodbye        Say goodbye

options:
  -h, --help       show this help message and exit
```

### 2. Hello Command

**Command:**
```bash
python simple_argparse.py hello Alice
```

**Output:**
```
Hello Alice
```

**Get hello command help:**
```bash
python simple_argparse.py hello --help
```

**Output:**
```
usage: simple_argparse.py hello [-h] name

positional arguments:
  name        Name to greet

options:
  -h, --help  show this help message and exit
```

### 3. Goodbye Command (Casual)

**Command:**
```bash
python simple_argparse.py goodbye Bob
```

**Output:**
```
Bye Bob!
```

### 4. Goodbye Command (Formal)

**Command:**
```bash
python simple_argparse.py goodbye Sarah --formal
```

**Output:**
```
Goodbye Ms. Sarah. Have a good day.
```

**Get goodbye command help:**
```bash
python simple_argparse.py goodbye --help
```

**Output:**
```
usage: simple_argparse.py goodbye [-h] [--formal] name

positional arguments:
  name      Name to say goodbye to

options:
  -h, --help  show this help message and exit
  --formal    Formal goodbye
```

## Understanding the Parts

### Required Arguments
- `name` - You must provide a name (no dashes needed)
- Example: `python simple_argparse.py hello Alice`

### Optional Flags  
- `--formal` - You can add this flag or leave it out (uses dashes)
- Example: `python simple_argparse.py goodbye Bob --formal`

### Commands
- `hello` - Says hello to someone
- `goodbye` - Says goodbye (casual or formal)

## Quick Reference

```bash
# Basic usage pattern
python simple_argparse.py [COMMAND] [NAME] [--FLAGS]

# Examples
python simple_argparse.py hello "John Doe"
python simple_argparse.py goodbye "Jane Smith" --formal
python simple_argparse.py --help
```

