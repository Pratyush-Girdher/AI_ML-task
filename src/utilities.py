import numpy as np
import pandas as pd
import pickle
from typing import Tuple

def train_test_split(X:np.ndarray, y:np.ndarray)->Tuple:
    n=X.shape[0]
    half=n//2
    X_train,X_test=X[:half],X[half:]
    y_train,y_test=y[:half],y[half:]
    return X_train, X_test, y_train, y_test

class StandardScalar:
    def __init__(self):
        self.mean_=None #will hold mean of colums
        self.scale_=None #will hold standard deviation of colums
    def fit(self,X:np.ndarray):
        self.mean_=np.mean(X,axis=0)
        self.scale_ = np.std(X,axis=0,ddof=0) # ddof =0 means we divide by N, instead of N-1
        self.scale_[self.scale_==0]=1.0 # if sandard deviation is 0 replace it by 1
    def transform(self, X:np.ndarray)->np.ndarray:
        return(X-self.mean_)/self.scale_
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        self.fit(X); return self.transform(X)
    def inverse_transform(self, X_scaled: np.ndarray) -> np.ndarray:
        return X_scaled * self.scale_ + self.mean_

def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return np.mean((y_true - y_pred)**2)
def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return np.mean(y_true == y_pred)

def save_model(obj, path: str):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)

def load_model(path: str):
    with open(path, 'rb') as f:
        return pickle.load(f)

