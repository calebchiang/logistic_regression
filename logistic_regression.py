import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('dataset.csv')
scaler = StandardScaler()


X_train = df.drop('TenYearCHD', axis=1).values
X_train_scaled = scaler.fit_transform(X_train)
y_train = df['TenYearCHD'].values

def sigmoid(z):
    g = 1 / (1 + np.exp(-z))
    return g

def predict(X, w, b):
    p = np.dot(X, w) + b
    return sigmoid(p)

def compute_cost(X, y, w, b):
    m = X.shape[0]
    f_wb = predict(X, w, b)
    cost = (-1 / m) * np.sum(y * np.log(f_wb) + (1 - y) * np.log(1 - f_wb))
    return cost

def compute_gradient(X, y, w, b):
    m = X.shape[0]
    f_wb = predict(X, w, b)
    error = f_wb - y
    dj_dw = (1 / m) * np.dot(X.T, error)
    dj_db = (1 / m) * np.sum(error)
    return dj_db, dj_dw

def gradient_descent(X, y, w, b, alpha, iterations):
    m = X.shape[0]
    cost_history = np.zeros(iterations)

    for i in range(iterations):
        dj_db, dj_dw = compute_gradient(X, y, w, b)
        w = w - alpha * dj_dw
        b = b - alpha * dj_db

        cost_history[i] = compute_cost(X, y, w, b)


        if i % 100 == 0:
            print(f"Iteration {i}: Cost = {cost_history[i]}")

    return w, b, cost_history


def main():
    w = np.array([ 0.26967541,  0.55870858,  0.03323484,  0.23385044,  0.03689789,  0.05246471,
                   0.08985292,  0.10942738,  0.07368041,  0.31774335, 0.02212815,  0.02286747,
                   0.00175421])
    b = -2.002094803823361


main()