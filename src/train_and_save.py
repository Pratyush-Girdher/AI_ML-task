import numpy as np
import os
import pandas as pd

from sklearn.datasets import make_classification, load_iris

from utilities import train_test_split, StandardScalar, save_model, mean_squared_error, accuracy

from linear_regression import LinearRegression
from Logistic_regression import LogisticRegression
from knn import Knn

MODELS_ROOT = "saved_models"
if not os.path.exists(MODELS_ROOT):
    os.makedirs(MODELS_ROOT)

def example_linear():
    print("Training Linear Regression")
    np.random.seed(0)
    n_samples=200
    X=np.random.randn(n_samples,3)
    true_w =np.array([1.5,-2.0,0.5])
    y = X @ true_w + 0.7 + np.random.randn(n_samples) * 0.5
    #splitting the data
    X_train, X_test, y_train, y_test= train_test_split(X,y)
    scaler = StandardScalar()
    #fitting the data
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    #training the model
    model = LinearRegression()
    model.fit(X_train_s, y_train)
    #predicting the model
    preds = model.predict(X_test_s)
    mse = mean_squared_error(y_test, preds)
    print(f"Linear Regression MSE: {mse:.4f}")
    #saving the model
    save_model({'model': model, 'scaler': scaler}, os.path.join(MODELS_ROOT, "linear_model.pkl"))
    print("Linear model saved.")

def example_logistic():
    print("Training Logistic Regression")
    # Using make_classification to generate a binary dataset
    X, y = make_classification(n_samples=400, n_features=4, n_informative=3, n_redundant=0, n_classes=2, random_state=1)
    
    #splitting the data
    X_train, X_test, y_train, y_test= train_test_split(X,y)
    scaler = StandardScalar()
    #fitting the data
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    #training the model
    model = LogisticRegression(lr=0.05, n_iter=1000)
    model.fit(X_train_s, y_train)
    #predicting the model
    preds = model.predict(X_test_s)
    acc = accuracy(y_test, preds)
    print(f"Logistic Regression accuracy: {acc:.4f}")
    #saving the model
    save_model({'model': model, 'scaler': scaler}, os.path.join(MODELS_ROOT, "logistic_model.pkl"))
    print("Logistic model saved.")

def example_knn():
    print("Training KNN ")
    #using load_iris to get multi class dataset
    iris = load_iris(as_frame=False)
    X, y = iris.data, iris.target    
    #splitting the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)
    scaler = StandardScalar()
    #fitting the data
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    #training the model
    model = Knn(k=5)
    model.fit(X_train_s, y_train)
    #predicting the model
    preds = model.predict(X_test_s)
    acc = accuracy(y_test, preds)
    print(f"KNN accuracy: {acc:.4f}")
    #saving the model
    save_model({'model': model, 'scaler': scaler}, os.path.join(MODELS_ROOT, "KNN_model.pkl"))
    print("KNN model saved.")

if __name__ == "__main__":
    example_linear()
    example_logistic()
    example_knn()
    
    print("\nAll models trained, evaluated, and saved successfully to saved_models/!")





