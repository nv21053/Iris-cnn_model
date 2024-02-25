from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np

# Load the trained model
model = tf.keras.models.load_model('iris_model.h5')

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the input data from the form
    sepal_length = float(request.form['sepal_length'])
    sepal_width = float(request.form['sepal_width'])
    petal_length = float(request.form['petal_length'])
    petal_width = float(request.form['petal_width'])

    # Preprocess the input data
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

    # Perform the prediction
    prediction_probabilities = model.predict(input_data)
    predicted_class_index = np.argmax(prediction_probabilities)
    
    # Get the predicted class label
    classes = ['Setosa', 'Versicolor', 'Virginica']
    predicted_class = classes[predicted_class_index]
    
    return render_template('result.html', predicted_class=predicted_class)

if __name__ == '__main__':
    app.run()