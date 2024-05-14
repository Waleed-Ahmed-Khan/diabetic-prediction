import os
import glob
import streamlit as st
import pandas as pd
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
import matplotlib.pyplot as plt
import seaborn as sns

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
    
    # Further code for integration...
    
if __name__ == "__main__":
    app()
