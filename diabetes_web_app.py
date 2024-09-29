from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Initialize Flask app
app = Flask(__name__)

# Load the pre-trained TensorFlow model
model = tf.keras.models.load_model('diabetes_tf_model.h5')

# Assuming you've saved the scaler used for feature scaling, load or re-initialize it
# If you haven't saved it, you'll need to use the same scaling process as used during training
scaler = StandardScaler()

# Define a function to preprocess the input data
def preprocess_data(input_data):
    # Convert input data to NumPy array and reshape for model input
    input_data = np.array(input_data).reshape(1, -1)
    
    # Standardize the input data using the pre-fitted scaler
    input_data = scaler.transform(input_data)
    return input_data

# Home route to display input form
@app.route('/')
def index():
    return render_template('index.html')  # This will render a simple input form

# Route to handle form submission and perform prediction
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the form data from the user input (all 8 feature values)
        features = [float(x) for x in request.form.values()]
        
        # Preprocess the input data
        processed_data = preprocess_data(features)
        
        # Make prediction using the loaded TensorFlow model
        prediction = model.predict(processed_data)
        
        # Rounding prediction (assuming binary classification with 0 or 1)
        prediction = int(np.round(prediction[0][0]))

        # Translate the prediction result into readable format
        result = "Positive for Diabetes" if prediction == 1 else "Negative for Diabetes"
        return render_template('result.html', prediction=result)
    
    except Exception as e:
        return str(e)

# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True)
