# Poetry Documentation

Poetry is a modern dependency management and packaging tool for Python that simplifies project management and virtual environment handling.

## Table of Contents
- [Installation](#installation)
- [Basic Commands](#basic-commands)
- [Project Management](#project-management)
- [Dependency Management](#dependency-management)
- [Virtual Environment](#virtual-environment)
- [Building and Publishing](#building-and-publishing)
- [Configuration](#configuration)
- [Useful Tips](#useful-tips)

## Installation

### macOS/Linux (Recommended)
```bash
curl -sSL https://install.python-poetry.org | python3 -
```

### Windows (PowerShell)
```powershell
(Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | py -
```

### Alternative Installation Methods

#### Using pip (not recommended for production)
```bash
pip install poetry
```

#### Using Homebrew (macOS)
```bash
brew install poetry
```

#### Using conda
```bash
conda install poetry
```

### Verify Installation
```bash
poetry --version
```



## Basic Commands

### Getting Help
```bash
# Show all available commands
poetry --help

# Get help for a specific command
poetry <command> --help
```

### Version Information
```bash
# Show Poetry version
poetry --version
```

## Project Management

### Creating a New Project
```bash
# Create a new project with basic structure
poetry new my-project

# Create a new project in current directory
poetry init
```

### Project Structure
When you create a new project, Poetry generates:
```
my-project/
├── pyproject.toml          # Project configuration
├── README.md              # Project documentation
├── my_project/            # Source code directory
│   └── __init__.py
└── tests/                 # Test directory
    └── __init__.py
```

### Initialize Existing Project
```bash
# Interactive initialization in existing directory
poetry init
```

## Dependency Management

### Adding Dependencies
```bash
# Add a dependency
poetry add requests

# Add a development dependency
poetry add pytest --group dev

# Add dependency with version constraint
poetry add "django>=3.0,<4.0"

# Add dependency from git repository
poetry add git+https://github.com/user/repo.git

# Add local dependency
poetry add ./my-local-package
```

### Removing Dependencies
```bash
# Remove a dependency
poetry remove requests

# Remove development dependency
poetry remove pytest --group dev
```

### Updating Dependencies
```bash
# Update all dependencies
poetry update

# Update specific dependency
poetry update requests

# Update to latest compatible versions
poetry update --lock
```

### Listing Dependencies
```bash
# Show installed packages
poetry show

# Show dependency tree
poetry show --tree

# Show outdated packages
poetry show --outdated

# Show only main dependencies
poetry show --only=main

# Show only development dependencies
poetry show --only=dev
```

### Installing Dependencies
```bash
# Install all dependencies from pyproject.toml
poetry install

# Install only main dependencies (exclude dev)
poetry install --only=main

# Install with extras
poetry install --extras "mysql redis"
```

## Virtual Environment

### Managing Virtual Environments
```bash
# Show virtual environment info
poetry env info

# Show virtual environment path
poetry env info --path

# List available environments
poetry env list

# Create new environment with specific Python version
poetry env use python3.9

# Remove virtual environment
poetry env remove python3.9
```

### Activating Virtual Environment
```bash
# Activate virtual environment (creates subshell)
poetry shell

# Run command in virtual environment
poetry run python script.py
poetry run pytest
poetry run jupyter notebook

# Exit virtual environment
exit  # (when in poetry shell)
```

## Useful Tips

### Working with pyproject.toml
The `pyproject.toml` file is the heart of your Poetry project:

```toml
[tool.poetry]
name = "my-project"
version = "0.1.0"
description = "A sample Python project"
authors = ["Your Name <you@example.com>"]

[tool.poetry.dependencies]
python = "^3.8"
requests = "^2.25.1"

[tool.poetry.group.dev.dependencies]
pytest = "^6.0"
black = "^21.0"
flake8 = "^3.8"

[build-system]
requires = ["poetry-core>=1.0.0"]
build-backend = "poetry.core.masonry.api"
```

### Lock File
- `poetry.lock` contains exact versions of all dependencies
- Always commit this file to version control
- Ensures reproducible builds across environments

### Export Requirements
```bash
# Export to requirements.txt format
poetry export -f requirements.txt --output requirements.txt

# Export development dependencies
poetry export -f requirements.txt --dev --output requirements-dev.txt

# Export without hashes (for compatibility)
poetry export -f requirements.txt --without-hashes
```


### Integration with IDEs
- Install `Even Better Toml` or `Better Toml` Extensions On `Vscode/Cursor` .

### Best Practices
1. Always use `poetry install` instead of `pip install` in Poetry projects
2. Commit both `pyproject.toml` and `poetry.lock` to version control
3. Use dependency groups to separate development and production dependencies
4. Pin Python version in `pyproject.toml` for consistency
5. Use `poetry shell` for interactive development
6. Use `poetry run` for running scripts and commands
