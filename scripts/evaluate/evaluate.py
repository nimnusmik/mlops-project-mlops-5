import numpy as np
import sys

def evaluate(model, val_loader, logger):
    total_loss = 0
    all_predictions = []
    correct = 0
    total = 0

    logger.write("모델 평가 시작...")  
    try:
        for features, labels in val_loader:
            predictions = model.forward(features)

            eps = 1e-8
            loss = -np.mean(np.sum(labels * np.log(predictions + eps), axis=1))
            total_loss += loss * len(features)  

            predicted = np.argmax(predictions, axis=1)
            true = np.argmax(labels, axis=1)

            all_predictions.extend(predicted)

            correct += np.sum(predicted == true)
            total += len(true)

        accuracy = correct / total if total > 0 else 0
        avg_loss = total_loss / len(val_loader.features) if len(val_loader.features) > 0 else 0 
        
        logger.write(f"모델 평가 완료. 손실: {avg_loss:.4f}, 정확도: {accuracy:.4f}")  
        return avg_loss, all_predictions, accuracy
    except Exception as e:
        logger.write(f"[ERROR] 모델 평가 중 오류 발생: {e}", print_error=True)  
        raise