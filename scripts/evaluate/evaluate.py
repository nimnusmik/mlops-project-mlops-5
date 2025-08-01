import numpy as np

def evaluate(model, val_loader):
    total_loss = 0
    all_predictions = []
    correct = 0
    total = 0

    for features, labels in val_loader:
        predictions = model.forward(features)

        # loss (cross-entropy)
        eps = 1e-8
        loss = -np.mean(np.sum(labels * np.log(predictions + eps), axis=1))
        total_loss += loss * len(features)

        predicted = np.argmax(predictions, axis=1)
        true = np.argmax(labels, axis=1)

        all_predictions.extend(predicted)

        correct += np.sum(predicted == true)
        total += len(true)

    accuracy = correct / total if total > 0 else 0
    return total_loss / len(val_loader), all_predictions, accuracy