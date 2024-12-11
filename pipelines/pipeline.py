import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from imblearn.over_sampling import SMOTE
import joblib

# Step 1: Load Data
def load_data(filepath):
    """Loads the dataset from the given file path."""
    return pd.read_csv(filepath)

# Step 2: Preprocess Data
def preprocess_data(data):
    """Preprocesses the data: handles missing values and feature scaling."""
    # Example: Dropping null values (if any)
    data = data.dropna()
    
    # Splitting features and target
    X = data.drop('Class', axis=1)
    y = data['Class']
    return X, y

# Step 3: Handle Imbalanced Data
def handle_imbalance(X, y):
    """Applies SMOTE to handle class imbalance."""
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    return X_resampled, y_resampled

# Step 4: Split Data
def split_data(X, y):
    """Splits the data into training and testing sets."""
    return train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Train Model
def train_model(X_train, y_train):
    """Trains a RandomForestClassifier on the training data."""
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    return model

# Step 6: Evaluate Model
def evaluate_model(model, X_test, y_test):
    """Evaluates the model on the test data."""
    y_pred = model.predict(X_test)
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("Accuracy Score:", accuracy_score(y_test, y_pred))

# Step 7: Save Model
def save_model(model, filepath):
    """Saves the trained model to the specified file path."""
    joblib.dump(model, filepath)

# Main Pipeline
if __name__ == "__main__":
    # File paths
    data_filepath = "data/raw/creditcard.csv"
    model_filepath = "models/credit_fraud_model.pkl"

    # Load and preprocess data
    print("Loading data...")
    data = load_data(data_filepath)
    X, y = preprocess_data(data)

    # Handle imbalance
    print("Handling class imbalance...")
    X_resampled, y_resampled = handle_imbalance(X, y)

    # Split data
    print("Splitting data...")
    X_train, X_test, y_train, y_test = split_data(X_resampled, y_resampled)

    # Train model
    print("Training model...")
    model = train_model(X_train, y_train)

    # Evaluate model
    print("Evaluating model...")
    evaluate_model(model, X_test, y_test)

    # Save model
    print("Saving model...")
    save_model(model, model_filepath)
    print(f"Model saved to {model_filepath}")
