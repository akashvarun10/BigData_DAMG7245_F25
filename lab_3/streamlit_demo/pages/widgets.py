# pages/widgets.py - Interactive widgets demonstration page
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, time, timedelta
import json

def show_widgets():
    """Interactive widgets demonstration page"""
    
    # Page header
    st.title("🎛️ Interactive Widgets Showcase")
    st.markdown("Explore all the interactive widgets available in Streamlit v1.50.0")
    st.markdown("---")
    
    # Basic Input Widgets
    st.header("📝 Basic Input Widgets")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Text Inputs")
        
        # Text input
        name = st.text_input("👤 Enter your name:", placeholder="John Doe")
        
        # Text area
        message = st.text_area(
            "💬 Enter a message:", 
            placeholder="Write your message here...",
            height=100
        )
        
        # Number input
        age = st.number_input("🎂 Enter your age:", min_value=0, max_value=120, value=25)
        
        # Password input (simulated with text_input)
        password = st.text_input("🔒 Password:", type="password")
        
    with col2:
        st.subheader("Selection Widgets")
        
        # Selectbox
        favorite_color = st.selectbox(
            "🎨 What's your favorite color?",
            ["Red", "Blue", "Green", "Yellow", "Purple", "Orange"]
        )
        
        # Multiselect
        hobbies = st.multiselect(
            "🎯 Select your hobbies:",
            ["Reading", "Sports", "Music", "Travel", "Gaming", "Cooking", "Art"],
            default=["Reading", "Music"]
        )
        
        # Radio buttons
        gender = st.radio(
            "⚧️ Gender:",
            ["Male", "Female", "Other", "Prefer not to say"],
            horizontal=True
        )
        
        # Checkbox
        newsletter = st.checkbox("📧 Subscribe to newsletter")
    
    # Display collected information
    if name:
        st.success(f"Hello {name}! 👋")
        st.json({
            "name": name,
            "age": age,
            "favorite_color": favorite_color,
            "hobbies": hobbies,
            "gender": gender,
            "newsletter_subscription": newsletter,
            "message": message[:50] + "..." if len(message) > 50 else message
        })
    
    st.markdown("---")
    
    # Slider and Range Widgets
    st.header("🎚️ Sliders and Ranges")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Numeric Sliders")
        
        # Regular slider
        temperature = st.slider(
            "🌡️ Temperature (°C):",
            min_value=-20,
            max_value=50,
            value=22,
            step=1
        )
        
        # Range slider
        price_range = st.slider(
            "💰 Price range ($):",
            min_value=0,
            max_value=1000,
            value=(100, 500),
            step=10
        )
        
        # Select slider
        size = st.select_slider(
            "👕 Clothing size:",
            options=["XS", "S", "M", "L", "XL", "XXL"],
            value="M"
        )
    
    with col2:
        st.subheader("Date and Time")
        
        # Date input
        birthday = st.date_input(
            "🎂 Your birthday:",
            value=datetime(1990, 1, 1)
        )
        
        # Time input
        meeting_time = st.time_input(
            "⏰ Meeting time:",
            value=time(14, 30)
        )
        
        # Color picker
        accent_color = st.color_picker("🎨 Pick an accent color:", "#FF6B6B")
    
    # Advanced Widgets
    st.markdown("---")
    st.header("🚀 Advanced Widgets")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Modern UI Elements")
        
        # Toggle
        dark_mode = st.toggle("🌙 Dark mode")
        
        # Pills (tags)
        selected_tags = st.pills(
            "🏷️ Select tags:",
            ["Python", "JavaScript", "Machine Learning", "Data Science", "Web Dev"],
            selection_mode="multi"
        )
        
        # Segmented control
        view_mode = st.segmented_control(
            "👁️ View mode:",
            ["Grid", "List", "Cards"],
            default="Grid"
        )
        
        # Feedback widget
        feedback = st.feedback("thumbs")
        if feedback is not None:
            st.write(f"Thanks for your feedback: {'👍' if feedback == 1 else '👎'}")
    
    with col2:
        st.subheader("File and Media Inputs")
        
        # File uploader (simulated)
        uploaded_file = st.file_uploader(
            "📁 Choose a file:",
            type=['csv', 'txt', 'pdf', 'xlsx']
        )
        
        if uploaded_file is not None:
            st.success(f"Uploaded: {uploaded_file.name}")
        
        # Camera input (placeholder)
        st.write("📸 Camera input:")
        if st.button("Take Photo"):
            st.info("📷 Camera functionality would work in a real deployment")
        
        # Audio input (placeholder)
        st.write("🎤 Audio input:")
        if st.button("Record Audio"):
            st.info("🎵 Audio recording functionality would work in a real deployment")
    
    # Interactive Forms
    st.markdown("---")
    st.header("📋 Forms and Actions")
    
    # Form example
    with st.form("user_profile_form", clear_on_submit=True):
        st.subheader("👤 User Profile Form")
        
        col1, col2 = st.columns(2)
        
        with col1:
            form_name = st.text_input("Full Name:")
            form_email = st.text_input("Email:")
            form_phone = st.text_input("Phone Number:")
        
        with col2:
            form_country = st.selectbox("Country:", ["USA", "Canada", "UK", "Germany", "France", "Other"])
            form_experience = st.slider("Years of experience:", 0, 20, 5)
            form_skills = st.multiselect("Skills:", ["Python", "SQL", "Machine Learning", "Data Visualization"])
        
        form_bio = st.text_area("Bio:", height=100)
        form_agree = st.checkbox("I agree to the terms and conditions")
        
        # Form submission
        submitted = st.form_submit_button("🚀 Submit Profile", use_container_width=True)
        
        if submitted:
            if form_name and form_email and form_agree:
                st.success("✅ Profile submitted successfully!")
                st.json({
                    "name": form_name,
                    "email": form_email,
                    "phone": form_phone,
                    "country": form_country,
                    "experience": form_experience,
                    "skills": form_skills,
                    "bio": form_bio
                })
            else:
                st.error("❌ Please fill in all required fields and agree to terms.")
    
    # Button Examples
    st.markdown("---")
    st.header("🔘 Button Examples")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Regular button
        if st.button("🎯 Primary Action", use_container_width=True):
            st.balloons()
            st.success("Primary action executed!")
    
    with col2:
        # Download button (simulated)
        sample_data = pd.DataFrame({
            'Column A': range(1, 11),
            'Column B': np.random.randn(10)
        })
        csv_data = sample_data.to_csv(index=False)
        
        st.download_button(
            label="📥 Download Sample Data",
            data=csv_data,
            file_name="sample_data.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    with col3:
        # Link button
        st.link_button(
            "🌐 Visit Streamlit Gallery",
            "https://streamlit.io/gallery",
            use_container_width=True
        )
    
    # Chat Interface Demo
    st.markdown("---")
    st.header("💬 Chat Interface Demo")
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hello! How can I help you today?"}
        ]
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Type your message here..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)
        
        # Add assistant response (simulated)
        response = f"Thanks for your message: '{prompt}'. This is a demo response!"
        st.session_state.messages.append({"role": "assistant", "content": response})
        with st.chat_message("assistant"):
            st.write(response)
    
    # Clear chat button
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = [
            {"role": "assistant", "content": "Hello! How can I help you today?"}
        ]
        st.rerun()
    
    # Widget State Summary
    st.markdown("---")
    st.header("📊 Widget State Summary")
    
    # Create summary of all widget states
    widget_summary = {
        "Basic Info": {
            "name": name if name else "Not provided",
            "age": age,
            "favorite_color": favorite_color
        },
        "Preferences": {
            "hobbies": hobbies,
            "size": size,
            "dark_mode": dark_mode,
            "view_mode": view_mode
        },
        "Settings": {
            "temperature": temperature,
            "price_range": price_range,
            "accent_color": accent_color,
            "newsletter": newsletter
        }
    }
    
    # Display in expandable sections
    for section, data in widget_summary.items():
        with st.expander(f"📋 {section}", expanded=False):
            st.json(data)
    
    st.markdown("---")
    st.info("💡 **Tip:** All widgets above are interactive! Try changing values and see how they update in real-time.")