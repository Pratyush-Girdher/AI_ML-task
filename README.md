Machine Learning Algorithms from Scratch
This repository contains fundamental machine learning algorithms implemented from scratch using Python and NumPy, adhering to the principles of building models without relying on high-level libraries like scikit-learn for the core modeling logic. We use sklearn.datasets exclusively for generating reliable synthetic test data.
Problem Statement
The goal of this task was to develop a deep understanding of the mathematical and algorithmic foundations of core machine learning models by implementing them using only NumPy for numerical computation. This exercise focuses on building the models themselves rather than relying on external libraries for core operations.

The implemented models are:

Linear Regression (for regression)

Logistic Regression (for binary classification)

K-Nearest Neighbors (KNN) (for multi-class classification)

Run Instructions
Prerequisites
You need NumPy, Pandas, and the data generation utilities from Scikit-learn installed.

Install dependencies:

```bash
pip install numpy pandas scikit-learn
```
Train all models:

```Bash

python src/train_and_save.py
```
