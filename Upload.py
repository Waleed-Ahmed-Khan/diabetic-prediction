import streamlit as st
import os
import numpy as np
from datetime import datetime
import pandas as pd
from streamlit_option_menu import option_menu as om
import plotly.express as px
import glob
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Input
import matplotlib.pyplot as plt
import seaborn as sns


page =om("Select Option",["Uploads", "EDA", "ML", "Prediction",], icons=['download', 'bi-pie-chart', 'puzzle', 'speedometer2'], menu_icon='cast', default_index =0, orientation='horizontal')
page
   

if page== 'Uploads':
    # Create a directory to save the uploaded files
    upload_directory = "uploaded_files"
    if not os.path.exists(upload_directory):
        os.makedirs(upload_directory)

    # Title of the Streamlit app
    st.title("Upload CSV File")

    # File uploader
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])
     
    st.markdown("If you dont have proper file or dont know about it please download it from kaggle by following link and then uplaod it after upload your file will process...")
   
    st.markdown("https://www.kaggle.com/datasets/mathchi/diabetes-data-set")
    st.markdown("Move to the EDA and other options if you dont want to upload your data as there is already pre upload data processed.")
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


elif page =='EDA':
        
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

    if __name__ == "__main__":
        app()

elif page =='ML':
        
    # Function to load the latest file
    def load_latest_file(folder_path):
        # Get the latest file in the folder
        files = glob.glob(os.path.join(folder_path, "*.csv"))
        latest_file = max(files, key=os.path.getmtime)
        return pd.read_csv(latest_file)

    # Function to train and evaluate various classification models
    def train_and_evaluate_models(data):
        # Separate features and target
        X = data.drop(columns=['Outcome'])
        y = data['Outcome']
        
        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale the features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        # Initialize a dictionary to store performance metrics
        metrics_dict = {}
        confusion_matrices = {}

        # Define the models
        models = {
            "Logistic Regression": LogisticRegression(random_state=42),
            "Decision Tree": DecisionTreeClassifier(random_state=42),
            "Random Forest": RandomForestClassifier(random_state=42),
            "Support Vector Machine": SVC(random_state=42),
            "Neural Network": Sequential([
                Dense(16, input_shape=(X_train.shape[1],), activation='relu'),
                Dropout(0.2),
                Dense(8, activation='relu'),
                Dropout(0.2),
                Dense(1, activation='sigmoid')  # Binary classification
            ])
        }

        # Train and evaluate each model
        for model_name, model in models.items():
            # Train the model
            if model_name == "Neural Network":
                # Compile the model
                model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
                # Train the neural network
                early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
                model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100, batch_size=32, callbacks=[early_stopping])
                # Predict with the neural network
                y_pred = model.predict(X_test).round()
            else:
                # Train the model
                model.fit(X_train, y_train)
                # Predict with the model
                y_pred = model.predict(X_test)
            
            # Evaluate the model and calculate performance metrics
            report = classification_report(y_test, y_pred, output_dict=True)
            # Adjust how metrics are stored in metrics_dict
            metrics_dict[model_name] = {
                'accuracy': report.get('accuracy', 0),  # Get 'accuracy' if it exists, else 0
                'precision': report.get('weighted avg', {}).get('precision', 0),
                'recall': report.get('weighted avg', {}).get('recall', 0),
                'f1-score': report.get('weighted avg', {}).get('f1-score', 0)
            }
            
            # Calculate confusion matrix
            confusion_matrices[model_name] = confusion_matrix(y_test, y_pred)
        
        # Display performance metrics
        st.subheader("Model Performance Comparison")
        metrics = ['accuracy', 'precision', 'recall', 'f1-score']

        # Create a dot plot for each metric
        for metric in metrics:
            metric_values = [metrics_dict[model_name][metric] for model_name in models.keys()]
            model_names = list(models.keys())
            st.subheader(f"{metric.capitalize()} Comparison")
            
            # Plot the dot plot
            fig, ax = plt.subplots()
            ax.scatter(metric_values, model_names, color='blue')
            ax.set_xlabel(metric.capitalize())
            ax.set_yticks(model_names)
            ax.set_yticklabels(model_names)
            
            # Display the plot
            st.pyplot(fig)
            
        # Display confusion matrices
        st.subheader("Confusion Matrices")
        for model_name, cm in confusion_matrices.items():
            st.subheader(f"{model_name} Confusion Matrix")
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_xlabel('Predicted')
            ax.set_ylabel('True')
            st.pyplot(fig)

    # Example usage in your Streamlit app
    def app():
        # Define the path to the uploaded files folder
        uploaded_folder_path = 'uploaded_files'
        st.text("Note: We Use 20% Data on Test and 80 percent on Training while having 42 random state")
        # Load the latest CSV file
        data = load_latest_file(uploaded_folder_path)
        
        # Train and evaluate models
        train_and_evaluate_models(data)

        
    if __name__ == "__main__":
        app()
        # Further code for integration...


elif page =='Prediction':

    # Load the latest CSV file from the uploaded_files folder
    def load_latest_file(folder_path):
        files = glob.glob(os.path.join(folder_path, "*.csv"))
        latest_file = max(files, key=os.path.getmtime)
        return pd.read_csv(latest_file)


    # Train and evaluate neural network model
    def train_neural_network(X_train, y_train, X_test, y_test):
        # Define the neural network model with an explicit Input layer
        model = Sequential([
            Input(shape=(X_train.shape[1],)),  # Explicitly specify the input shape
            Dense(16, activation='relu'),
            Dropout(0.2),
            Dense(8, activation='relu'),
            Dropout(0.2),
            Dense(1, activation='sigmoid')  # Binary classification
        ])
        
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        
        # Early stopping callback
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        
        # Train the model with the training data
        model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100, batch_size=32, callbacks=[early_stopping])
        
        # Make predictions on the test data
        y_pred = model.predict(X_test).round()
        
        # Generate classification report and confusion matrix
        report = classification_report(y_test, y_pred, output_dict=True)
        cm = confusion_matrix(y_test, y_pred)
        
        # Return the trained model, accuracy, report, and confusion matrix
        return model, report['accuracy'], report, cm


    # Train and evaluate random forest model
    def train_random_forest(X_train, y_train, X_test, y_test):
        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        report = classification_report(y_test, y_pred, output_dict=True)
        cm = confusion_matrix(y_test, y_pred)
        
        return model, report['accuracy'], report, cm

    # Train and evaluate logistic regression model
    def train_logistic_regression(X_train, y_train, X_test, y_test):
        model = LogisticRegression(random_state=42)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        report = classification_report(y_test, y_pred, output_dict=True)
        cm = confusion_matrix(y_test, y_pred)
        
        return model, report['accuracy'], report, cm

    # Train and evaluate SVM model
    def train_svm(X_train, y_train, X_test, y_test):
        model = SVC(random_state=42)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        report = classification_report(y_test, y_pred, output_dict=True)
        cm = confusion_matrix(y_test, y_pred)
        
        return model, report['accuracy'], report, cm

    # Train and evaluate decision tree model
    def train_decision_tree(X_train, y_train, X_test, y_test):
        model = DecisionTreeClassifier(random_state=42)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        report = classification_report(y_test, y_pred, output_dict=True)
        cm = confusion_matrix(y_test, y_pred)
        
        return model, report['accuracy'], report, cm

    # Dictionary to map each algorithm to its respective function
    algorithm_functions = {
        'Neural Network': train_neural_network,
        'Random Forest': train_random_forest,
        'Logistic Regression': train_logistic_regression,
        'SVM': train_svm,
        'Decision Tree': train_decision_tree,
    }

    def app():
        # Define the path to the uploaded files folder
        uploaded_folder_path = 'uploaded_files'
        
        # Load the latest CSV file
        data = load_latest_file(uploaded_folder_path)
        
        # Separate features and target
        X = data.drop(columns=['Outcome'])
        y = data['Outcome']
        
        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale the features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        # Provide the user with a dropdown to select an algorithm
        selected_algorithm = st.selectbox("Select an algorithm", list(algorithm_functions.keys()))
        
        # Get the function for the selected algorithm
        selected_function = algorithm_functions[selected_algorithm]
        
        # Train and evaluate the selected algorithm
        model, accuracy, report, cm = selected_function(X_train, y_train, X_test, y_test)
        
        # Display the accuracy of the selected algorithm
        st.write(f"Accuracy of {selected_algorithm}: {accuracy:.2f}")
        
        # Display the classification report
    # Display the classification report in tabular form
        st.write(f"Classification Report for {selected_algorithm}:")
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df)

        
        # Display the confusion matrix
        st.write(f"Confusion Matrix for {selected_algorithm}:")
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, cmap='Blues', fmt='d', ax=ax)
        st.pyplot(fig)
        
        # Define input fields for user data
        user_input = {
            "Pregnancies": st.number_input("Pregnancies", value=0),
            "Glucose": st.number_input("Glucose", value=0),
            "BloodPressure": st.number_input("BloodPressure", value=0),
            "SkinThickness": st.number_input("SkinThickness", value=0),
            "Insulin": st.number_input("Insulin", value=0),
            "BMI": st.number_input("BMI", min_value=0.0,  # Set minimum value
            max_value=100.0,  # Set maximum value (you can adjust as needed)
            value=0.0,      # Default value
            step=0.1       # Step size to allow rational input
            ),
            "DiabetesPedigreeFunction": st.number_input("DiabetesPedigreeFunction",  min_value=0.0,  # Set minimum value
            max_value=2.0,  # Set maximum value (you can adjust as needed)
            value=0.5,      # Default value
            step=0.01       # Step size to allow rational input
            ),
        
            "Age": st.number_input("Age", value=0)
        }
        
        # Use the trained model to predict
        prediction = model.predict(scaler.transform(pd.DataFrame([user_input]))).round()
        
        # Display the prediction result
        if prediction == 1:
            st.write("This model predicts that the person **has diabetes**.")
           
            
        else:
            st.write("This model predicts that the person **does not have diabetes**.")
            
            # Run the app
    if __name__ == "__main__":
      app()


