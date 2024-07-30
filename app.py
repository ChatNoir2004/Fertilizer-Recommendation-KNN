from flask import Flask, render_template, request
import pickle
import numpy as np

# Define the KNNModel class
class KNNModel:
    def __init__(self, k=3):
        self.k = k

    def fit(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train

    def euclidean_distance(self, a, b):
        return np.sqrt(np.sum((a - b) ** 2))

    def predict(self, x_test):
        predictions = []
        for test_point in x_test:
            distances = np.array([self.euclidean_distance(test_point, train_point) for train_point in self.x_train])
            nearest_indices = np.argsort(distances)[:self.k]
            nearest_labels = self.y_train[nearest_indices]
            unique, counts = np.unique(nearest_labels, return_counts=True)
            predicted_label = unique[np.argmax(counts)]
            predictions.append(predicted_label)
        return np.array(predictions)

app = Flask(__name__)

# Load the k-NN model and the labels
with open('knn_fertilizer_model.pkl', 'rb') as file:
    data = pickle.load(file)
    knn_model = data['model']
    unique_labels = data['labels']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Retrieve and convert input data
        Nitrogen = request.form.get('Nitrogen', '0')
        Potassium = request.form.get('Potassium', '0')
        Phosphorous = request.form.get('Phosphorous', '0')

        # Ensure the data is numerical
        try:
            Nitrogen = float(Nitrogen)
            Potassium = float(Potassium)
            Phosphorous = float(Phosphorous)
        except ValueError:
            return render_template('index.html', result="Error: Please enter valid numerical values.")

        # Create input array for prediction
        input_data = np.array([[Nitrogen, Phosphorous, Potassium]])

        # Predict using the k-NN model
        result_encoded = knn_model.predict(input_data)
        fertilizer_name = unique_labels[result_encoded[0]]

    except Exception as e:
        result = f"Error: {str(e)}"
        return render_template('index.html', result=result)

    return render_template('index.html', result=fertilizer_name)

if __name__ == '__main__':
    app.run(debug=True)
