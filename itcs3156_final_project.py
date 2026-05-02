#imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#loading the database
df = pd.read_csv("Womens Clothing E-Commerce Reviews.csv")
print(df.head())
print(df.shape)
print(df.columns)
print(df.info())
df = df.dropna(subset = ["Rating", "Department Name", "Class Name"])
df["Review Text"] = df["Review Text"].fillna("")

#target variables
df["Popular"] = (df["Rating"] >= 4).astype(int)
print(df["Popular"].value_counts())

#features
features = [
    "Age",
    "Positive Feedback Count",
    "Department Name",
    "Class Name"
]
X = df[features]
y = df["Popular"]
#encoding data
X = pd.get_dummies(X, columns=["Department Name", "Class Name"])

#normalizing and training
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

#linear regression (gradient descent)!!!!
#bias term
X_train_b = np.c_[np.ones((X_train.shape[0], 1)), X_train]
X_test_b = np.c_[np.ones((X_test.shape[0], 1)), X_test]

#initializing weights
np.random.seed(42)
weights = np.random.randn(X_train_b.shape[1])

#gradient descent function
def gradient_descent(X, y, weights, learning_rate = 0.01, epochs = 1000):
    n = len(y)
    loss_history = []
    for epoch in range(epochs):
        y_pred = X.dot(weights)
        error = y_pred - y
        gradient = (2/n) * X.T.dot(error)
        weights = weights - learning_rate * gradient
        loss = (1/n) * np.sum(error **2)
        loss_history.append(loss)

        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss:.4f}")
    return weights, loss_history

#training gd
weights, loss_history = gradient_descent(X_train_b, y_train.values, weights)
y_pred = X_test_b.dot(weights)
y_pred_class = (y_pred >= 0.5).astype(int)


#the perceptron!!!!
X_train_p = X_train
#X_train_p = X_test
y_train_p = y_train.values
y_test_p = y_test.values

#initializing weights
np.random.seed(42)
weights = np.random.randn(X_train_p.shape[1])
bias = 0

#perceptron function
def perceptron_train(X, y, weights, bias, learning_rate = 0.01, epochs = 100):
    for epoch in range(epochs):
        errors = 0
        for i in range(len(X)):
            linear_output = np.dot(X[i], weights) + bias 
            y_pred = 1 if linear_output >= 0 else 0
            update = learning_rate * (y[i] - y_pred)
            weights += update *X[i]
            bias += update
            if update != 0:
                errors += 1
        print(f"Epoch {epoch}, Misclassifications: {errors}")
    return weights, bias

#training perceptron
weights_p, bias_p = perceptron_train(X_train_p, y_train_p, weights, bias)
def perceptron_predict(X, weights, bias):
    linear_output = np.dot(X, weights) + bias
    return (linear_output >= 0).astype(int)
y_pred_p = perceptron_predict(X_test, weights_p, bias_p)

#display of raw data
#rating distribution
plt.hist(df["Rating"], color="#ff69b4", bins=5)
plt.title("Distribution of Ratings")
plt.xlabel("Rating")
plt.ylabel("Frequency")
plt.show()
#category vs average rating
df.groupby("Department Name")["Rating"].mean().plot(kind="bar", color="#ff69b4")
plt.title("Average Rating by Department")
plt.ylabel("Average Rating")
plt.show()
#feedback vs rating
plt.scatter(df["Positive Feedback Count"], df["Rating"], color="#ff69b4")
plt.title("Feedback Count vs Rating")
plt.xlabel("Positive Feedback Count")
plt.ylabel("Rating")
plt.show()
#age distribution
plt.hist(df["Age"], bins=20, color="#ff69b4")
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.show()

#evaluations
#gd
accuracy = accuracy_score(y_test, y_pred_class)
print("Gradient descent accuracy:", accuracy)

plt.plot(loss_history, color = "#ff69b4")
plt.xlabel("Epoch")
plt.ylabel("Loss (MSE)")
plt.title("Gradient Descent Learning Curve")
plt.show()

#perceptron
accuracy_p = accuracy_score(y_test, y_pred_p)
print("Perceptron Accuracy:", accuracy_p)

plt.bar(["GD", "Perceptron"], [accuracy, accuracy_p], color = ["#ff69b4", "#8e44ad"])
plt.title("Model Comparison")
plt.ylabel("Accuracy")
plt.show()