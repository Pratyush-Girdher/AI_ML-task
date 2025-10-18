import numpy as np

class LinearRegression:
    def __init__(self, lr=0.01, n_iter=1000):
        self.lr = lr
        self.n_iter= n_iter
        self.w=None
        self.b=None
    def fit(self, X,y):
        m,n = X.shape
        self.w=np.zeros(n)
        self.b=0

        for i in range(self.n_iter):
            y_pred = X@self.w +self.b

            error = y_pred - y

            dw= (2/m)*(X.T@error)
            db =(2/m)*np.sum(error)

            self.w-=self.lr*dw
            self.b-=self.lr*db

    def predict(self,X):
        return X@self.w +self.b

    