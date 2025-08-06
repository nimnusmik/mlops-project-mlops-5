import os
import numpy as np
import datetime
import pickle
import sys  

from scripts.utils.utils import model_dir, save_hash  


class MoviePredictor:
    name = "movie_predictor"

    def __init__(self, input_dim, hidden_dim, num_classes):
        
        self.weights1 = np.random.randn(input_dim, hidden_dim) * np.sqrt(2 / input_dim)
        self.bias1 = np.zeros((1, hidden_dim))

        self.weights2 = np.random.randn(hidden_dim, hidden_dim) * np.sqrt(2 / hidden_dim)
        self.bias2 = np.zeros((1, hidden_dim))

        self.weights3 = np.random.randn(hidden_dim, num_classes) * np.sqrt(2 / hidden_dim)
        self.bias3 = np.zeros((1, num_classes))

    def relu(self, x):
        return np.maximum(0, x)

    def batch_norm(self, x):
        mean = np.mean(x, axis=0, keepdims=True)
        std = np.std(x, axis=0, keepdims=True) + 1e-8
        return (x - mean) / std

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def forward(self, x):
        self.z1 = np.dot(x, self.weights1) + self.bias1
        self.a1 = self.relu(self.z1)
        self.a1 = self.batch_norm(self.a1)

        self.z2 = np.dot(self.a1, self.weights2) + self.bias2
        self.a2 = self.relu(self.z2)
        self.a2 = self.batch_norm(self.a2)

        self.z3 = np.dot(self.a2, self.weights3) + self.bias3
        self.output = self.softmax(self.z3)
        return self.output

    def backward(self, x, y, output, lr=0.001):
        m = len(x)

        # Cross-entropy gradient
        dz3 = (output - y) / m
        dw3 = np.dot(self.a2.T, dz3)
        db3 = np.sum(dz3, axis=0, keepdims=True)

        da2 = np.dot(dz3, self.weights3.T)
        dz2 = da2 * (self.z2 > 0)
        dw2 = np.dot(self.a1.T, dz2)
        db2 = np.sum(dz2, axis=0, keepdims=True)

        da1 = np.dot(dz2, self.weights2.T)
        dz1 = da1 * (self.z1 > 0)
        dw1 = np.dot(x.T, dz1)
        db1 = np.sum(dz1, axis=0, keepdims=True)

        # Update
        self.weights3 -= lr * dw3
        self.bias3 -= lr * db3
        self.weights2 -= lr * dw2
        self.bias2 -= lr * db2
        self.weights1 -= lr * dw1
        self.bias1 -= lr * db1

    def load_state_dict(self, state_dict):
        self.weights1 = state_dict["weights1"]
        self.bias1 = state_dict["bias1"]
        self.weights2 = state_dict["weights2"]
        self.bias2 = state_dict["bias2"]
        self.weights3 = state_dict["weights3"]
        self.bias3 = state_dict["bias3"]


def model_save(model, model_params, epoch, loss, scaler, label_encoder, logger): 
    save_dir = model_dir(model.name)
    os.makedirs(save_dir, exist_ok=True)

    current_time = datetime.datetime.now().strftime("%y%m%d%H%M%S")
    save_path = os.path.join(save_dir, f"E{epoch}_T{current_time}.pkl")

    save_data = {
        "epoch": epoch,
        "model_params": model_params,
        "model_state_dict": {
            "weights1": model.weights1,
            "bias1": model.bias1,
            "weights2": model.weights2,
            "bias2": model.bias2,
            "weights3": model.weights3,
            "bias3": model.bias3,
        },
        "loss": loss,
        "scaler": scaler,
        "label_encoder": label_encoder,
    }

    with open(save_path, "wb") as f:
        pickle.dump(save_data, f)

    logger.write(f"모델 저장 완료: {save_path}") 
    save_hash(save_path, logger)  
    
    return save_path
