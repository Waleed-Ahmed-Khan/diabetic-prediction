import streamlit as st
import os
from datetime import datetime
import pandas as pd

# Create a directory to save the uploaded files
upload_directory = "uploaded_files"
if not os.path.exists(upload_directory):
    os.makedirs(upload_directory)

# Title of the Streamlit app
st.title("Upload CSV File")

# File uploader
uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

if uploaded_file is not None:
    # Get the current date and time
    current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Define the file path using the current date and time
    file_path = os.path.join(upload_directory, f"data_{current_datetime}.csv")

    # Save the uploaded file
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Provide feedback to the user
    st.success(f"File uploaded successfully and saved as {file_path}")

    # Load the data using pandas
    data = pd.read_csv(file_path)
    
    # Display the first few rows of the data
    st.write("Here are the first few rows of the data:")
    st.dataframe(data.head())
