import numpy as np
from collections import Counter
from typing import List

class Knn:
    def __init__(self,k:int = 5):
        self.k=k
        self.X_train=None
        self.y_train=None

    def fit(self,X:np.ndarray,y:np.ndarray):
        self.X_train = X
        self.y_train = y
        
    def _euclidean_distance(self,x1:np.ndarray,x2:np.ndarray)->float:
        return np.sqrt(np.sum((x1-x2)**2))
    
    def _predict_single_sample(self, x_test:np.ndarray)->int:
        distances: List[float]=[
            self._euclidean_distance(x_test,x_train)
            for x_train in self.X_train
        ]

        k_indices:np.ndarray=np.argsort(distances)[:self.k]

        k_nearest_labes: np.ndarray = self.y_train[k_indices]

        most_common = Counter(k_nearest_labes).most_common(1)

        return most_common[0][0]
    
    def predict(self,X_test: np.ndarray)->np.ndarray:
        predictions = [self._predict_single_sample(x) for x in X_test]
        
        return np.array(predictions)
