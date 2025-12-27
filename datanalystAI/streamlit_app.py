import streamlit as st
import os
from pathlib import Path
from agent.analysis import analyze_data

# Page Configuration
st.set_page_config(page_title="Data Analysis Agent", layout="wide")

# Ensure directories exist
os.makedirs('uploads', exist_ok=True)
os.makedirs('output', exist_ok=True)

st.title("Data Analysis Agent")

# Sidebar/Form for inputs
with st.sidebar:
    st.header("Settings")
    uploaded_file = st.file_uploader("Upload Data File", type=["csv", "xlsx", "json"])
    description = st.text_area("Data Description")
    n_value = st.number_input("N (parameter)", min_value=1, value=2)
    analyze_button = st.button("Start Analysis")

if analyze_button:
    if uploaded_file is not None:
        # Save uploaded file temporarily
        file_path = os.path.join('uploads', uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        with st.spinner('Analyzing data...'):
            try:
                # Call your model function
                # Note: Ensure analyze_data returns valid paths
                html_path, zip_path = analyze_data(
                    file_path=file_path, 
                    data_description=description, 
                    n=n_value
                )

                # Display the HTML Report
                with open(html_path, 'r', encoding='utf-8') as f:
                    html_content = f.read()
                
                st.success("Analysis Complete!")
                
                # Provide Download Link for the ZIP file
                with open(zip_path, "rb") as fp:
                    st.download_button(
                        label="Download Analysis Results (ZIP)",
                        data=fp,
                        file_name=os.path.basename(zip_path),
                        mime="application/zip"
                    )

                # Render the HTML content
                st.components.v1.html(html_content, height=800, scrolling=True)

            except Exception as e:
                st.error(f"An error occurred: {e}")
    else:
        st.error("Please upload a file first.")