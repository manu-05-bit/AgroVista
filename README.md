🌱 Crop & Plant Disease Prediction System

This project is a Flask-based web application that predicts the most suitable crop to grow based on soil and weather conditions, and also detects plant diseases from uploaded leaf images using a deep learning model. Additionally, it features a simple e-commerce system for agricultural products.

🚀 Features

Crop Recommendation

Uses Random Forest Classifier trained on Crop.csv.

Inputs: Nitrogen (N), Phosphorus (P), Potassium (K), Temperature, Humidity, pH, Rainfall.

Outputs: Best crop recommendation.

Plant Disease Prediction

CNN model (plant_disease_model.keras) trained on leaf image dataset.

Detects common diseases (e.g., Bacterial Wilt, Early Blight, Leaf Mold, etc.).

Shows predicted class with confidence score.

E-commerce Module

Product listing with images and prices.

Add to cart, remove from cart, checkout, and order confirmation.

🛠️ Tech Stack

Frontend: HTML, CSS, Bootstrap, Jinja2 Templates

Backend: Flask (Python)

Machine Learning: scikit-learn (Random Forest, Decision Tree)

Deep Learning: TensorFlow/Keras (CNN for disease detection)

Database: CSV-based dataset (Crop.csv)

Other Tools: NumPy, Pandas, OpenCV

📂 Project Structure
├── app.py                  # Flask main app (routes for crop & disease prediction, e-commerce)
├── crop_prediction.py       # ML pipeline for crop prediction (Decision Tree)
├── predict.py               # ML pipeline with Random Forest (crop accuracy evaluation)
├── start.py                 # CNN model training for plant disease prediction
├── Crop.csv                 # Dataset for crop recommendation
├── DATA/                    # Training dataset for plant disease detection
├── templates/               # HTML templates (index, crop form, result, e-commerce, etc.)
├── static/                  # Static files (CSS, JS, product images, uploads)
└── plant_disease_model.keras # Saved CNN model

⚙️ Installation

Clone the repository

git clone https://github.com/yourusername/crop-disease-prediction.git
cd crop-disease-prediction


Create a virtual environment

python -m venv venv
source venv/bin/activate   # On Mac/Linux
venv\Scripts\activate      # On Windows


Install dependencies

pip install -r requirements.txt


Run the app

python app.py


Open in browser: http://127.0.0.1:5000/

📊 Dataset

Crop.csv → Crop recommendation data (NPK, weather, pH, etc.).

DATA/ → Plant leaf image dataset for training CNN model.

🎯 Future Enhancements

Improve accuracy with advanced ML/DL models.

Add support for more crop datasets.

Deploy app on Heroku / AWS / Azure.

Connect e-commerce to a real payment gateway.

👨‍💻 Author

Developed by Vara Devamani 🌟
Feel free to contribute via Pull Requests.
