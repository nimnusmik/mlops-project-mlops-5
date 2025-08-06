import math
import sys
import numpy as np


class SimpleDataLoader: 
    def __init__(self, features, labels, logger, batch_size=16, shuffle=True):
        self.logger = logger
        self.features = features
        self.labels = labels
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_samples = len(features)
        self.indices = np.arange(self.num_samples)
        #DataLoader 초기화 로그 메세지
        self.logger.write(f"SimpleDataLoader initialized with {self.num_samples} samples, batch_size={self.batch_size}, shuffle={self.shuffle}.")

    def __iter__(self):
        if self.shuffle:
            np.random.shuffle(self.indices)
        self.current_idx = 0
        return self

    def __next__(self):
        if self.current_idx >= self.num_samples:
            raise StopIteration

        start_idx = self.current_idx
        end_idx = start_idx + self.batch_size
        self.current_idx = end_idx

        batch_indices = self.indices[start_idx:end_idx]
        return self.features[batch_indices], self.labels[batch_indices]
        
    def __len__(self):
        return math.ceil(self.num_samples / self.batch_size)