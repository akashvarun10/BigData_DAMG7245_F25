import argparse

def main():
    # Create parser
    parser = argparse.ArgumentParser(description="Simple CLI with argparse")
    
    # Add subcommands
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Hello command
    hello_parser = subparsers.add_parser('hello', help='Say hello')
    hello_parser.add_argument('name', help='Name to greet')
    
    # Goodbye command  
    goodbye_parser = subparsers.add_parser('goodbye', help='Say goodbye')
    goodbye_parser.add_argument('name', help='Name to say goodbye to')
    goodbye_parser.add_argument('--formal', action='store_true', help='Formal goodbye')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Handle commands
    if args.command == 'hello':
        print(f"Hello {args.name}")
    
    elif args.command == 'goodbye':
        if args.formal:
            print(f"Goodbye Ms. {args.name}. Have a good day.")
        else:
            print(f"Bye {args.name}!")
    
    else:
        # No command provided
        parser.print_help()

if __name__ == "__main__":
    main()

# Get help
#python simple_argparse.py --help

# Hello command
#python simple_argparse.py hello Alice
# Output: Hello Alice

# Goodbye command (casual)
#python simple_argparse.py goodbye Bob  
# Output: Bye Bob!

# Goodbye command (formal)
#python simple_argparse.py goodbye Sarah --formal

