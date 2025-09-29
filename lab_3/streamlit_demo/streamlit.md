# Streamlit Multi-Page Demo App 📊

## 📁 Project Structure

```
streamlit-demo-app/
│
├── main.py                 # Main application entry point with navigation
├── pages/                  # Page modules directory
│   ├── greeting.py       # Sample Greeting
│   ├── widgets.py         # Interactive widgets demonstration
├── requirements.txt       # Python dependencies
└── README.md             # This documentation file
```

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.8 or higher
- pip (Python package installer)

### Step-by-Step Installation

1. **Clone or download the project files**
   ```bash
   # Create project directory
   mkdir streamlit-demo-app
   cd streamlit-demo-app
   ```

2. **Create the directory structure**
   ```bash
   mkdir pages
   touch pages/__init__.py
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Create all the Python files**
   - Copy the provided code into their respective files:
     - `main.py` - Main application file
     - `pages/dashboard.py` - Dashboard page
     - `pages/widgets.py` - Widgets page  
     - `pages/data_analysis.py` - Data analysis page

## 🚀 Running the Application

### Local Development
```bash
streamlit run main.py
```

The app will open in your default web browser at `http://localhost:8501`

### Alternative Streamlit Commands
```bash
# Show Streamlit version
streamlit version

# Clear cache
streamlit cache clear

# Show configuration
streamlit config show

# Access Streamlit documentation
streamlit docs
```

### Streamlit Documentation
- [Official Streamlit Documentation](https://docs.streamlit.io)
- [Streamlit API Reference](https://docs.streamlit.io/library/api-reference)
- [Streamlit Gallery](https://streamlit.io/gallery)


## 🐛 Troubleshooting

### Common Issues

1. **Import Errors**
   ```bash
   ModuleNotFoundError: No module named 'pages'
   ```
   **Solution**: Ensure you have `__init__.py` in the `pages/` directory

2. **Navigation Not Working**
   ```bash
   AttributeError: module 'streamlit' has no attribute 'navigation'
   ```
   **Solution**: Upgrade to Streamlit v1.50.0 or higher
   ```bash
   pip install streamlit>=1.50.0 --upgrade
   ```

3. **Performance Issues**
   - Clear Streamlit cache: `streamlit cache clear`
   - Enable caching decorators for expensive operations
   - Use pagination for large datasets

4. **Layout Issues**
   - Check column ratios in `st.columns()`
   - Ensure proper container hierarchy
   - Use `use_container_width=True` for responsive components

## 🚀 Deployment

### Streamlit Community Cloud
1. Push code to GitHub repository
2. Connect at [share.streamlit.io](https://share.streamlit.io)
3. Deploy with one click

### Docker
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "main.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

### Local Network Access
```bash
streamlit run main.py --server.address=0.0.0.0
```
