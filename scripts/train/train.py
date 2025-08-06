import numpy as np
from tqdm import tqdm
import sys


def train(model, train_loader, logger):
    total_loss = 0

    logger.write("모델 학습 시작...")
    try:
        for features, labels in tqdm(train_loader, desc="학습 중", leave=False): 
            predictions = model.forward(features)

            # cross-entropy loss
            eps = 1e-8
            loss = -np.mean(np.sum(labels * np.log(predictions + eps), axis=1))

            model.backward(features, labels, predictions, lr=0.001)

            total_loss += loss
        
        avg_loss = total_loss / len(train_loader)
        logger.write(f"모델 학습 완료. 평균 손실: {avg_loss:.4f}")  
        return avg_loss
    except Exception as e:
        logger.write(f"[ERROR] 모델 학습 중 오류 발생: {e}", print_error=True) 
        raise  