import numpy as np
import pickle
from flask import Flask, render_template, request
from sklearn.datasets import load_iris

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get input values from form
    features = [float(request.form.get(f)) for f in ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
    data = np.array(features).reshape(1, -1)

    # Load model and predict
    with open('knn_model.pkl', 'rb') as f:
        model = pickle.load(f)

    prediction = model.predict(data)[0]

    # FIX: Load the dataset properly to get target_names
    iris = load_iris()
    flower_name = iris.target_names[prediction]

    return render_template('index.html', prediction_text=f'The predicted flower is: {flower_name.title()}')

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)
