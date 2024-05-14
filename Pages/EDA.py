import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import os
from datetime import datetime

# Function to load the data
def load_data(file_path):
    return pd.read_csv(file_path)

# Function to clean the data (handle missing values and other anomalies)
def clean_data(df):
    # Fill missing values with mean (or any other strategy you prefer)
    df.fillna(df.mean(), inplace=True)
    return df

# Function to normalize the data
def normalize_data(df):
    # Normalize numerical columns using Min-Max scaling
    numeric_cols = df.select_dtypes(include=np.number).columns
    df[numeric_cols] = (df[numeric_cols] - df[numeric_cols].min()) / (df[numeric_cols].max() - df[numeric_cols].min())
    return df

# Function to perform EDA
def perform_eda(df):
    st.header("Exploratory Data Analysis (EDA)")
    
    # Display summary statistics
    st.subheader("Summary Statistics")
    st.write(df.describe())
    
    # Option for selecting columns
    st.subheader("Select Columns for Analysis")
    columns = df.columns
    # Provide unique keys for each selectbox widget
    col1 = st.selectbox("Select First Column", columns, key='col1_select')
    col2 = st.selectbox("Select Second Column", columns, key='col2_select')
    
    # Validate selected columns for line chart
    if col1 == col2:
        st.error("Selected columns for line chart must be different.")
    else:
        # Display line chart
        st.subheader("Line Chart")
        if col1 and col2:
            # If both columns are numerical
            if df[col1].dtype == np.number and df[col2].dtype == np.number:
                # Use aggregation to reduce the number of data points
                df_agg = df.groupby(col1).agg({col2: 'mean'}).reset_index()
                fig = px.line(df_agg, x=col1, y=col2, title=f"Line Chart: {col2} vs {col1}")
                st.plotly_chart(fig)

  
    # Display bar chart
    st.subheader("Bar Chart")
    if col1 and col2 and df[col2].dtype == np.number:
        # Display bar chart with counts
        fig = px.bar(df, x=col1, y=col2, title=f"Bar Chart: {col2} vs {col1}")
        fig.update_traces(text=fig.data[0].y)
        st.plotly_chart(fig)

    # Display correlations
    st.subheader("Correlations")
    corr = df.corr()
    fig = px.imshow(corr, title="Correlation Matrix", color_continuous_scale='RdBu_r')
    st.plotly_chart(fig)
    
    # Data Distribution: Histograms and Box Plots
    st.subheader("Data Distribution")
    for col in df.select_dtypes(include=np.number).columns:
        st.subheader(f"Histogram and Box Plot of {col}")
        fig_hist = px.histogram(df, x=col, title=f"Histogram of {col}")
        st.plotly_chart(fig_hist)
        fig_box = px.box(df, y=col, title=f"Box Plot of {col}")
        st.plotly_chart(fig_box)
    
    # Pairwise Relationships: Scatter Plots and Pair Plots
    st.subheader("Pairwise Relationships")
    if col1 and col2:
        fig_scatter = px.scatter(df, x=col1, y=col2, title=f"Scatter Plot: {col2} vs {col1}")
        st.plotly_chart(fig_scatter)
        
    # Missing Values
    st.subheader("Missing Values")
    missing_values = df.isnull().sum()
    st.write("Missing values in each column:")
    st.write(missing_values)
    
    # Heatmap of missing values
    st.subheader("Heatmap of Missing Values")
    fig_missing = px.imshow(df.isnull(), title="Missing Values Heatmap")
    st.plotly_chart(fig_missing)

    # Categorical Analysis: Grouping by categorical variables
    if col1 and df[col1].dtype in ['object', 'category'] and col2 in df.columns:
        st.subheader(f"Categorical Analysis: Grouping {col2} by {col1}")
        grouped_data = df.groupby(col1)[col2].mean().reset_index()
        fig_bar = px.bar(grouped_data, x=col1, y=col2, title=f"Bar Chart: {col2} vs {col1}")
        st.plotly_chart(fig_bar)
    
    # Feature Engineering (You can add more as per your data and requirements)


# Function to get the latest file from the uploaded_files directory
def get_latest_file(directory):
    # Get the list of files in the directory
    files = [os.path.join(directory, f) for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    # Sort the files by modification time and get the latest file
    latest_file = max(files, key=os.path.getmtime)
    return latest_file

# Main application code
def app():
    # Title of the Streamlit app
    st.title("Exploratory Data Analysis (EDA)")
    
    # Directory containing the uploaded files
    upload_directory = "uploaded_files"
    
    # Get the latest file from the directory
    latest_file = get_latest_file(upload_directory)
    
    if latest_file:
        # Load the data
        data = load_data(latest_file)
        
        # Clean the data
        data_clean = clean_data(data)
        
        # Normalize the data
        data_normalized = normalize_data(data_clean)
        
        # Perform EDA
        perform_eda(data_normalized)
    else:
        st.warning("No uploaded files found. Please upload a CSV file first.")

# Run the app
if __name__ == "__main__":
    app()
