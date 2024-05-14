import os
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Input
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import seaborn as sns
import matplotlib.pyplot as plt
import glob


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
        st.write(prediction)
        
    else:
        st.write("This model predicts that the person **does not have diabetes**.")
        st.write(prediction)
# Main execution
if __name__ == "__main__":
    app()
