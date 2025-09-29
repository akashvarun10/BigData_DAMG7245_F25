# main.py - Main Streamlit Application with Navigation
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time

# Page configuration - must be first Streamlit command
st.set_page_config(
    page_title="Streamlit Demo App",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import page functions
from pages.greeting import greeting_page
from pages.widgets import show_widgets

def main():
    """Main application with navigation"""
    
    # Create navigation
    pg = st.navigation([
        st.Page(greeting_page, title="📊 Greeting", url_path="dashboard", default=True),
        st.Page(show_widgets, title="🎛️ Interactive Widgets", url_path="widgets"),
    ], position="top")
    
    # Run the selected page
    pg.run()

if __name__ == "__main__":
    main()

