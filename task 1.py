import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tkinter as tk
from tkinter import ttk, messagebox

class NeuralNetworkTask:
    def __init__(self, eta=0.005, epochs=200, mse_threshold=0.01, use_bias=True, algorithm="Perceptron"):
        self.eta = eta
        self.epochs = epochs
        self.mse_threshold = mse_threshold
        self.use_bias = use_bias
        self.algorithm = algorithm
        self.weights = None
        self.bias = None
        self.scaler = StandardScaler()

    def load_data(self, filepath, feature1, feature2, class1, class2):
        data = pd.read_csv(filepath)
        class_column = 'bird category'
        
        # Filter classes
        data = data[(data[class_column] == class1) | (data[class_column] == class2)]
        
        # Select features and labels
        X = data[[feature1, feature2]].values
        y = np.where(data[class_column] == class1, -1, 1)
        
        # Standardize features
        X = self.scaler.fit_transform(X)
        
        # Split into train and test sets with a fixed sample size for each class
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=40, stratify=y)
        return X_train, X_test, y_train, y_test

    def initialize_parameters(self, n_features):
        self.weights = np.random.rand(n_features)
        self.bias = np.random.rand(1)[0] if self.use_bias else 0.0

    def activation_function(self, x):
        return np.where(x >= 0, 1, -1)

    def train(self, X, y):
        n_features = X.shape[1]
        self.initialize_parameters(n_features)
        mse_history = []

        for epoch in range(self.epochs):
            errors = []
            for xi, target in zip(X, y):
                if self.algorithm == "Perceptron":
                    output = self.activation_function(np.dot(xi, self.weights) + self.bias)
                    error = target - output
                    self.weights += self.eta * error * xi
                    if self.use_bias:
                        self.bias += self.eta * error

                elif self.algorithm == "Adaline":
                    output = np.dot(xi, self.weights) + self.bias
                    error = target - output
                    self.weights += self.eta * error * xi
                    if self.use_bias:
                        self.bias += self.eta * error
                    errors.append(error ** 2)

            if self.algorithm == "Adaline":
                mse = np.mean(errors)
                mse_history.append(mse)
                if mse < self.mse_threshold:
                    print(f"Stopping early at epoch {epoch} with MSE={mse}")
                    break

    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        return self.activation_function(linear_output)

    def plot_decision_boundary(self, X, y):
        plt.scatter(X[y == -1][:, 0], X[y == -1][:, 1], color='blue', label='Class -1')
        plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], color='orange', label='Class 1')
        
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        x_values = np.linspace(x_min, x_max, 100)
        
        if self.weights[1] != 0:
            y_values = -(self.weights[0] * x_values + self.bias) / self.weights[1]
            plt.plot(x_values, y_values, color='black', linestyle='-', linewidth=2, label='Decision Boundary')

        plt.xlabel("Feature 1 (Standardized)")
        plt.ylabel("Feature 2 (Standardized)")
        plt.legend()
        plt.title(f"{self.algorithm} Decision Boundary")
        plt.show()

    def evaluate(self, X_test, y_test):
        y_pred = self.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        accuracy = accuracy_score(y_test, y_pred)
        print("Confusion Matrix:\n", cm)
        print("Accuracy:", accuracy)
        return cm, accuracy

# Tkinter GUI setup
def run_model():
    try:
        eta = float(eta_entry.get())
        epochs = int(epochs_entry.get())
        mse_threshold = float(mse_threshold_entry.get())
        use_bias = bias_var.get()
        algorithm = algo_var.get()
        feature1 = feature1_var.get()
        feature2 = feature2_var.get()
        class1 = class1_var.get()
        class2 = class2_var.get()
        
        # Initialize the neural network task
        nn_task = NeuralNetworkTask(eta=eta, epochs=epochs, mse_threshold=mse_threshold, use_bias=use_bias, algorithm=algorithm)
        
        # Load data, train, plot, and evaluate
        X_train, X_test, y_train, y_test = nn_task.load_data("birds.csv", feature1, feature2, class1, class2)
        nn_task.train(X_train, y_train)
        nn_task.plot_decision_boundary(X_train, y_train)
        nn_task.evaluate(X_test, y_test)
        
    except Exception as e:
        messagebox.showerror("Error", str(e))

root = tk.Tk()
root.title("Neural Network Task")

# GUI elements for inputs
tk.Label(root, text="Learning Rate (eta):").grid(row=0, column=0)
eta_entry = tk.Entry(root)
eta_entry.grid(row=0, column=1)

tk.Label(root, text="Epochs:").grid(row=1, column=0)
epochs_entry = tk.Entry(root)
epochs_entry.grid(row=1, column=1)

tk.Label(root, text="MSE Threshold:").grid(row=2, column=0)
mse_threshold_entry = tk.Entry(root)
mse_threshold_entry.grid(row=2, column=1)

bias_var = tk.BooleanVar()
tk.Checkbutton(root, text="Add Bias", variable=bias_var).grid(row=3, column=0, columnspan=2)

tk.Label(root, text="Algorithm:").grid(row=4, column=0)
algo_var = tk.StringVar(value="Perceptron")
tk.Radiobutton(root, text="Perceptron", variable=algo_var, value="Perceptron").grid(row=4, column=1)
tk.Radiobutton(root, text="Adaline", variable=algo_var, value="Adaline").grid(row=4, column=2)

feature1_var = tk.StringVar()
feature2_var = tk.StringVar()
class1_var = tk.StringVar()
class2_var = tk.StringVar()

tk.Label(root, text="Select Feature 1:").grid(row=5, column=0)
feature1_menu = ttk.Combobox(root, textvariable=feature1_var, values=["body_mass", "beak_length", "beak_depth", "fin_length"])
feature1_menu.grid(row=5, column=1)

tk.Label(root, text="Select Feature 2:").grid(row=6, column=0)
feature2_menu = ttk.Combobox(root, textvariable=feature2_var, values=["body_mass", "beak_length", "beak_depth", "fin_length"])
feature2_menu.grid(row=6, column=1)

tk.Label(root, text="Select Class 1:").grid(row=7, column=0)
class1_menu = ttk.Combobox(root, textvariable=class1_var, values=["A", "B", "C"])
class1_menu.grid(row=7, column=1)

tk.Label(root, text="Select Class 2:").grid(row=8, column=0)
class2_menu = ttk.Combobox(root, textvariable=class2_var, values=["A", "B", "C"])
class2_menu.grid(row=8, column=1)

run_button = tk.Button(root, text="Run Model", command=run_model)
run_button.grid(row=9, column=0, columnspan=2)

root.mainloop()
