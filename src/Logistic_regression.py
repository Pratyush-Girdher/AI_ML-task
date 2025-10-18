import numpy as np
from typing import Tuple
class LogisticRegression:
    def __init__(self, lr=0.05, n_iter=1000):
        self.lr = lr
        self.n_iter= n_iter
        self.w=None
        self.b=None

    def _sigmoid(self,z:np.ndarray)->np.ndarray:
        return 1/(1+np.exp(-z))
    
    def fit(self, X,y):
        m,n = X.shape
        self.w=np.zeros(n)
        self.b=0

        for i in range(self.n_iter):
            z = X@self.w +self.b
            y_pred = self._sigmoid(z)
            error = y_pred - y

            dw= (1/m)* np.dot(X.T,error)
            db =(1/m)*np.sum(error)

            self.w-=self.lr*dw
            self.b-=self.lr*db

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        z = X @ self.w + self.b
        return self._sigmoid(z)

    def predict(self, X: np.ndarray) -> np.ndarray:
        probabilities = self.predict_proba(X)
        return (probabilities >= 0.5).astype(int)

    