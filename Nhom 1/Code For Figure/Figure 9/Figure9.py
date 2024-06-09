import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Feature scaling
def feature_scaling(X):
    return (X - np.mean(X)) / np.std(X)

# Polynomial feature combination
def capacity(X, degree):
    m = X.shape[0]
    X_combined = np.ones((m, 1))  # Start with a column of ones for the bias term
    for d in range(1, degree + 1):
        X_combined = np.hstack((X_combined, X**d))
    return X_combined

# Analytical solution using the normal equation
def formular(X_combined, Y):
    return np.linalg.inv(X_combined.T.dot(X_combined)).dot(X_combined.T).dot(Y)

# Function to create animation
def draw2(X_train, X_test, Y_train, Y_test, x0_gd, x_min, x_max, y_min, y_max, x_padding, y_padding, features_count, eta=0.01):
    fig, ax = plt.subplots(figsize=(8, 6))

    def update(degree):
        ax.clear()
        ax.set_xlim(x_min - x_padding, x_max + x_padding)
        ax.set_ylim(y_min - y_padding, y_max + y_padding)
        
        # Combine features for polynomial regression of given degree
        X_train_combined = capacity(X_train, degree)
        X_test_combined = capacity(X_test, degree)
        x0_gd_combined = capacity(x0_gd, degree)

        # Combine features for optimal polynomial regression (degree = 5)
        X_train_combined_optimal = capacity(X_train, 5)
        X_test_combined_optimal = capacity(X_test, 5)
        x0_gd_combined_optimal = capacity(x0_gd, 5)

        # Calculate theta using normal equation
        theta = formular(X_train_combined, Y_train)
        theta_optimal = formular(X_train_combined_optimal, Y_train)

        # Calculate predicted Y values
        Y_train_predict = X_train_combined.dot(theta)
        Y_test_predict = X_test_combined.dot(theta)
        Y_predict = x0_gd_combined.dot(theta)

        Y_train_predict_optimal = X_train_combined_optimal.dot(theta_optimal)
        Y_test_predict_optimal = X_test_combined_optimal.dot(theta_optimal)
        Y_predict_optimal = x0_gd_combined_optimal.dot(theta_optimal)

        # Calculate errors
        train_error = mean_squared_error(Y_train, Y_train_predict)
        test_error = mean_squared_error(Y_test, Y_test_predict)
        train_error_optimal = mean_squared_error(Y_train, Y_train_predict_optimal)
        test_error_optimal = mean_squared_error(Y_test, Y_test_predict_optimal)

        # Plot the predicted regression line
        ax.plot(X_train, Y_train, 'ro', label='Train Data')
        ax.plot(X_test, Y_test, 'bo', label='Test Data')
        ax.plot(x0_gd, Y_predict, label=f'Predicted (Degree {degree})')
        ax.plot(x0_gd, Y_predict_optimal, label='Optimal Capacity', linestyle='--')
        ax.set_xlabel('Scaled X')
        ax.set_ylabel('Y')
        ax.legend(loc='upper right')
        ax.set_title(f'Polynomial Degree {degree}')
        ax.grid(True)

        # Display errors
        # ax.text(0.02, 0.95, f'Train Error: {train_error:.2f}', transform=ax.transAxes, fontsize=12, verticalalignment='top')
        # ax.text(0.02, 0.90, f'Test Error: {test_error:.2f}', transform=ax.transAxes, fontsize=12, verticalalignment='top')
        # ax.text(0.02, 0.85, f'Error Difference: {abs(train_error - test_error):.2f}', transform=ax.transAxes, fontsize=12, verticalalignment='top')
        textstr = f'Train Error {train_error:.2f}\nTest Error: {test_error:.2f}\nError Difference: {abs(train_error - test_error):.2f}'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)


    anim = FuncAnimation(fig, update, frames=np.arange(1, features_count+1), repeat=False, interval=1000)

    anim.save('underfittings.gif', writer='pillow', dpi=80)
    plt.show()

# Data
X = np.array([[33, 45, 49, 43, 51, 59, 67, 63, 69, 58, 34, 32, 20, 22, 26, 76, 80, 62, 55, 48, 44, 92, 100, 105, 110, 86, 87, 114, 103, 96, 131, 140, 139, 134, 162, 132, 150, 153, 162, 117, 125, 175, 184, 174, 174, 151, 146, 190, 183, 200, 191, 191, 211, 213, 202, 203, 216, 204, 221, 237, 215, 233, 232, 253, 242, 239, 250, 251, 260, 263, 256, 240, 226, 229, 237, 243, 251, 294, 271, 268, 281, 289, 266, 253]]).T
Y = np.array([[17, 33, 26, 21, 46, 37, 57, 52, 46, 46, 26, 7, 15, 7, 23, 67, 60, 70, 60, 53, 48, 58, 46, 39, 33, 61, 52, 24, 29, 39, 37, 47, 26, 20, 48, 30, 50, 40, 69, 15, 15, 67, 83, 94, 86, 75, 67, 60, 56, 50, 52, 76, 72, 63, 69, 81, 52, 37, 34, 64, 28, 50, 26, 33, 26, 16, 0, 17, 0, -13, -17, 0, 17, 5, -10, -25, -36, -57, -44, -55, -63, -35, -28, -64]]).T

features_count = 5

# Feature scaling
X_scaled = feature_scaling(X)

# Split the data into training and testing sets (7:3 ratio)
X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.3, random_state=42)

# Calculate axis limits based on the data
x_min, x_max = X_scaled.min(), X_scaled.max()
y_min, y_max = Y.min(), Y.max()

# Add padding around min and max values to avoid edge clipping
x_padding = (x_max - x_min) * 0.1
y_padding = (y_max - y_min) * 0.1

# Generate x0_gd range for predictions
x0_gd = np.linspace(x_min - x_padding, x_max + x_padding, 100).reshape(-1, 1)

# Run the draw2 function
draw2(X_train, X_test, Y_train, Y_test, x0_gd, x_min, x_max, y_min, y_max, x_padding, y_padding, features_count)
