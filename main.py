import numpy as np
import pickle
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend communication

# Load the trained model once when the server starts
MODEL_PATH = 'digitrec-model-v3.pkl'

try:
    with open(MODEL_PATH, 'rb') as f:
        W1, b1, W2, b2 = pickle.load(f)
    print("[INFO] Model successfully loaded.")
except FileNotFoundError:
    print("[ERROR] Model file not found! Ensure 'trained_model.pkl' exists.")
    W1, b1, W2, b2 = None, None, None, None


def ReLU(Z):
    return np.maximum(Z, 0)

def softmax(Z):
    exp_Z = np.exp(Z - np.max(Z))  # Stability trick to prevent overflow
    return exp_Z / np.sum(exp_Z, axis=0, keepdims=True)

def forward_prop(W1, b1, W2, b2, X):
    """Performs forward propagation through the neural network."""
    Z1 = W1.dot(X) + b1
    A1 = ReLU(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = softmax(Z2)
    return A2

def predict(X):
    """Predicts the digit from the input image vector."""
    if W1 is None or W2 is None:
        return {"error": "Model not loaded."}

    X = np.array(X).reshape(784, 1) / 255.0  # Normalize input
    A2 = forward_prop(W1, b1, W2, b2, X)
    prediction = np.argmax(A2, axis=0)[0]
    return int(prediction)


@app.route('/get_data', methods=['POST'])
def get_data():
    """Receives image data from frontend and predicts the digit."""
    try:
        data = request.get_json()
        pixel_data = data.get("pixelData", "")

        if not pixel_data:
            return jsonify({"error": "No data received."}), 400

        # Convert string of numbers into a NumPy array
        pixel_list = list(map(int, pixel_data.split(',')))

        if len(pixel_list) != 784:
            return jsonify({"error": "Invalid input size."}), 400

        prediction = predict(pixel_list)
        return jsonify({"prediction": prediction})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
