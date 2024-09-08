from flask import Flask, request, jsonify, render_template
import joblib
import tensorflow as tf
import numpy as np

app = Flask(__name__)

# Load model and scaler
model = tf.keras.models.load_model('model/model_datafull.h5')
model2 = tf.keras.models.load_model('model/model_dataaug.h5')

scaler = joblib.load('model/scaler.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get selected model
        model_name = request.form.get('model')
        
        # Load the appropriate model
        if model_name == 'model_datafull':
            model = tf.keras.models.load_model('model/model_datafull.h5')
        elif model_name == 'model_dataaug':
            model = tf.keras.models.load_model('model/model_dataaug.h5')
        else:
            return jsonify({'error': 'Invalid model selected'})

        # Get input data from the form
        input_data = [
            float(request.form.get('NT')),
            float(request.form.get('NR')),
            float(request.form.get('PM')),
            float(request.form.get('BF')),
            float(request.form.get('CUM')),
            float(request.form.get('ND')),
            float(request.form.get('Distance'))
        ]

        # Scale input data
        input_array = np.array([input_data])
        scaled_input = scaler.transform(input_array)

        # Make prediction
        prediction = model.predict(scaled_input)
        result = round(float(prediction[0][0]), 2)  # Round to 2 decimal places

        return jsonify({'prediction': f'{result:.2f} mm/s'})
    except Exception as e:
        return jsonify({'error': str(e)})


if __name__ == '__main__':
    app.run(debug=True)
