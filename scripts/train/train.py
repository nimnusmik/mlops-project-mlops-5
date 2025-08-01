import numpy as np
from tqdm import tqdm


def train(model, train_loader):
    total_loss = 0
    for features, labels in tqdm(train_loader, desc="Training", leave=False):
        predictions = model.forward(features)

        # cross-entropy loss
        eps = 1e-8
        loss = -np.mean(np.sum(labels * np.log(predictions + eps), axis=1))

        model.backward(features, labels, predictions, lr=0.001)

        total_loss += loss

    return total_loss / len(train_loader)