import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import joblib

# Function to load and preprocess the dataset
def load_and_preprocess_data(file_path):
    data = pd.read_csv(file_path)
    le = LabelEncoder()
    data['soil_type'] = le.fit_transform(data['soil_type'])
    data['crop'] = le.fit_transform(data['crop'])
    return data, le

# Function to split the dataset
def split_data(data):
    X = data[['temperature', 'humidity', 'soil_type']]
    y = data['crop']
    return train_test_split(X, y, test_size=0.2, random_state=42)

# Function to train the model
def train_model(X_train, y_train):
    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)
    return model

# Function to evaluate the model
def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Function to save the model and encoder
def save_model_and_encoder(model, encoder, model_path='crop_prediction_model.pkl', encoder_path='label_encoder.pkl'):
    joblib.dump(model, model_path)
    joblib.dump(encoder, encoder_path)

# Main function to run the entire process
def main():
    data, le = load_and_preprocess_data('Crop.csv')
    X_train, X_test, y_train, y_test = split_data(data)
    model = train_model(X_train, y_train)
    evaluate_model(model, X_test, y_test)
    save_model_and_encoder(model, le)

if __name__ == "__main__":
    main()
