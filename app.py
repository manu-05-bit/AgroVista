from flask import Flask, render_template, request, jsonify, redirect, url_for, make_response
import json
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from tensorflow.keras.models import load_model  # type: ignore
from tensorflow.keras.preprocessing import image    # type: ignore
import numpy as np
import os

app = Flask(__name__)

# Sample data for products
product_categories = [
    {"id": 0, "image": "/static/images/ph1.jpg", "name": "UREA-NEAM COATED", "price": 400},
    {"id": 1, "image": "/static/images/ph2.jpg", "name": "ROCK PHOSPHATE", "price":  356},
    {"id": 2, "image": "/static/images/ph3.jpg", "name": "ORGANIC GREENDROP", "price": 300},
    {"id": 3, "image": "/static/images/ph4.jpg", "name": "THRIPAN", "price": 250},
    {"id": 4, "image": "/static/images/ph5.jpg", "name": "MULTICLEAR", "price": 463},
    {"id": 5, "image": "/static/images/ph6.jpg", "name": "LAMBDA CYHALOTHRIN", "price": 180},
    {"id": 6, "image": "/static/images/ph7.jpg", "name": "HERBICIDE ALLOY", "price": 430},
    {"id": 7, "image": "/static/images/ph8.jpg", "name": "NATRAJ-CARBOXIN", "price": 360},
    {"id": 8, "image": "/static/images/ph9.jpg", "name": "TRIM-CAVIET EC", "price": 456},
    {"id": 9, "image": "/static/images/ph10.jpg", "name": "BIOFINISH-BIO PESTICIDE", "price": 254},
    {"id": 10, "image": "/static/images/ph11.jpg", "name": "PROFEX SUPER INSECTICIDE", "price": 276},
    {"id": 11, "image": "/static/images/ph12.jpg", "name": "METARHIZIUM ANISOPLIAE", "price": 350},
    {"id": 12, "image": "/static/images/ph13.jpg", "name": "VESTIGE AGRI", "price": 430},
]

# Load dataset and train the crop prediction model once
data = pd.read_csv('Crop.csv')
label1 = data.iloc[:, 7]
label_encoder = LabelEncoder()
encoded_crops = label_encoder.fit_transform(label1)
X = data.iloc[:, 0:7]
y = encoded_crops
X_train1, X_test, y_train1, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
X_train, X_val, y_train, y_val = train_test_split(X_train1, y_train1, test_size=0.2, random_state=2022)
RF = RandomForestClassifier()
RF.fit(X_train, y_train)





# Route to serve static files
@app.route('/static/<path:path>')
def send_static(path):
    return send_from_directory('static', path)  # type: ignore

# Load the trained model
try:
    model = load_model('plant_disease_model.keras')  # Ensure your model path is correct and there's no extra space
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# Preprocess the uploaded image
def preprocess_image(img_path):
    try:
        img = image.load_img(img_path, target_size=(150, 150))  # Ensure target size matches model input size
        img_array = image.img_to_array(img)
        img_array = img_array / 255.0  # Normalize the image (assuming model was trained with normalized inputs)
        img_array = np.expand_dims(img_array, axis=0)  # Expand dims to match model input shape (1, 150, 150, 3)
        return img_array
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return None

# Make predictions using the preprocessed image
def predict_disease(img_path):
    img_array = preprocess_image(img_path)
    if img_array is not None:
        prediction = model.predict(img_array)
        return prediction
    return None

# Routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/crop_prediction', methods=['GET', 'POST'])
def crop_prediction():
    if request.method == 'POST':
        N = float(request.form['N'])
        P = float(request.form['P'])
        K = float(request.form['K'])
        temperature = float(request.form['temperature'])
        humidity = float(request.form['humidity'])
        ph = float(request.form['ph'])
        rainfall = float(request.form['rainfall'])

        prediction_data = [[N, P, K, temperature, humidity, ph, rainfall]]
        prediction = RF.predict(prediction_data)
        predicted_crop = label_encoder.inverse_transform(prediction)[0]

        return render_template('crop_result.html', predicted_crop=predicted_crop)
    return render_template('crop_prediction.html')

@app.route('/disease_prediction', methods=['GET', 'POST'])
def disease_prediction():
    return render_template('disease_prediction.html')

# Handle image uploads and predictions
@app.route('/predict', methods=['POST'])  # Changed from '/result' to '/predict'
def predict():
    if request.method == 'POST':
        if 'file' not in request.files:
            return "No file part"
        img = request.files['file']
        if img.filename == '':
            return 'No selected file'
        if img:
            # Create directory if it doesn't exist
            if not os.path.exists('uploads'):
                os.makedirs('uploads')
            img_path = os.path.join('uploads', img.filename)
            img.save(img_path)

            # Perform prediction
            prediction = predict_disease(img_path)
            if prediction is None:
                return 'Error in prediction'

            # Define class labels (adjust these to match your model's output)
            class_labels = ['Bacterial Wilt', 'Early Blight', 'Leaf Mold', 
                            'Leaf Rust', 'Root Knot Nematodes', 
                            'Septoria Leaf Spot', 'Smut Disease', 'Healthy', 'Mosaic Virus']


            predicted_class_index = np.argmax(prediction)
            predicted_label = class_labels[predicted_class_index]

            # Show confidence level (probability of the predicted class)
            confidence = prediction[0][predicted_class_index] * 100
            confidence = round(confidence, 2)

            return render_template('result.html', prediction=predicted_label, confidence=confidence)


@app.route('/e_commerce')
def e_commerce():
    return render_template('e_commerce.html')

@app.route('/get_products')
def get_products():
    return jsonify(product_categories)

@app.route('/add_to_cart', methods=['POST'])
def add_to_cart():
    product_id = request.json.get('id')
    cart = request.cookies.get('cart')

    if cart:
        cart = json.loads(cart)
    else:
        cart = {}

    if product_id in cart:
        cart[product_id]['quantity'] += 1
    else:
        product = next((p for p in product_categories if p['id'] == product_id), None)
        if product:
            cart[product_id] = {**product, 'quantity': 1}

    response = make_response(jsonify({'success': True}))
    response.set_cookie('cart', json.dumps(cart))
    return response

@app.route('/del_from_cart', methods=['POST'])
def del_from_cart():
    product_id = request.json.get('id')
    cart = request.cookies.get('cart')

    if cart:
        cart = json.loads(cart)
    else:
        cart = {}

    if product_id in cart:
        del cart[product_id]

    response = make_response(jsonify({'success': True}))
    response.set_cookie('cart', json.dumps(cart))
    return response

@app.route('/get_cart')
def get_cart():
    cart = request.cookies.get('cart')

    if cart:
        cart = json.loads(cart)
    else:
        cart = {}

    cart_items = [product_categories[int(i)] for i in cart.keys()]
    return jsonify(cart_items)

@app.route('/shipping', methods=['GET', 'POST'])
def shipping():
    if request.method == 'POST':
        return redirect(url_for('order_confirmation'))
    return render_template('shipping.html')

@app.route('/order_confirmation')
def order_confirmation():
    return render_template('order_confirmation.html')

if __name__ == '__main__':
    app.run(debug=True)
