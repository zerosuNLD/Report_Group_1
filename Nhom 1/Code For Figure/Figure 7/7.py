import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

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

# Function to calculate mean squared error
def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

# Function to draw polynomial regression with given degree
def draw_polynomial_regression(X_train_scaled, Y_train, X_test_scaled, Y_test, x0_gd, degree, x_min, x_max, y_min, y_max, x_padding, y_padding):
    # Combine features for polynomial regression of given degree
    X_train_combined = capacity(X_train_scaled, degree)
    X_test_combined = capacity(X_test_scaled, degree)
    x0_gd_combined = capacity(x0_gd, degree)

    # Calculate theta using normal equation
    theta = formular(X_train_combined, Y_train)

    # Calculate predicted Y values
    Y_train_predict = X_train_combined.dot(theta)
    Y_test_predict = X_test_combined.dot(theta)
    Y_predict = x0_gd_combined.dot(theta)

    # Calculate training and testing errors
    train_error = mean_squared_error(Y_train, Y_train_predict)
    test_error = mean_squared_error(Y_test, Y_test_predict)

    # Plot the predicted regression line
    plt.plot(X_train_scaled, Y_train, 'ro', label='Training Data')
    plt.plot(X_test_scaled, Y_test, 'bo', label='Testing Data')
    plt.plot(x0_gd, Y_predict, label=f'Predicted (Degree {degree})')

    # Set plot limits and labels
    plt.xlim(x_min - x_padding, x_max + x_padding)
    plt.ylim(y_min - y_padding, y_max + y_padding)
    plt.xlabel('Scaled X')
    plt.ylabel('Y')

    # Ensure the legend is in the upper left corner
    plt.legend(loc='upper left')

    # Set title based on the degree
    if degree == 1:
        plt.title('Low Capacity', fontsize=14)
    elif degree == 5:
        plt.title('Optimal Capacity', fontsize=14)
    elif degree == 15:
        plt.title('High Capacity', fontsize=14)
    
    plt.grid(True)

    # Add text annotations for training and testing errors in the bottom right corner
    error_text = f'Train Error: {train_error:.2f}\nTest Error: {test_error:.2f}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    plt.text(0.95, 0.05, error_text, transform=plt.gca().transAxes, fontsize=10,
             verticalalignment='bottom', horizontalalignment='right', bbox=props)

# Data
X = np.array([[33, 45, 49, 43, 51, 59, 67, 63, 69, 58, 34, 32, 20, 22, 26, 76, 80, 62, 55, 48, 44, 92, 100, 105, 110, 86, 87, 114, 103, 96, 131, 140, 139, 134, 162, 132, 150, 153, 162, 117, 125, 175, 184, 174, 174, 151, 146, 190, 183, 200, 191, 191, 211, 213, 202, 203, 216, 204, 221, 237, 215, 233, 232, 253, 242, 239, 250, 251, 260, 263, 256, 240, 226, 229, 237, 243, 251, 294, 271, 268, 281, 289, 266, 253]]).T
Y = np.array([[17, 33, 26, 21, 46, 37, 57, 52, 46, 46, 26, 7, 15, 7, 23, 67, 60, 70, 60, 53, 48, 58, 46, 39, 33, 61, 52, 24, 29, 39, 37, 47, 26, 20, 48, 30, 50, 40, 69, 15, 15, 67, 83, 94, 86, 75, 67, 60, 56, 50, 52, 76, 72, 63, 69, 81, 52, 37, 34, 64, 28, 50, 26, 33, 26, 16, 0, 17, 0, -13, -17, 0, 17, 5, -10, -25, -36, -57, -44, -55, -63, -35, -28, -64]]).T

# Split data into training and testing sets (70% training, 30% testing)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

# Feature scaling
X_train_scaled = feature_scaling(X_train)
X_test_scaled = feature_scaling(X_test)

# Degrees for the polynomial regression
degrees = [1, 5, 15]

# Calculate axis limits based on the data
x_min, x_max = X_train_scaled.min(), X_train_scaled.max()
y_min, y_max = Y.min(), Y.max()

# Add padding around min and max values to avoid edge clipping
x_padding = (x_max - x_min) * 0.1
y_padding = (y_max - y_min) * 0.1

# Generate x0_gd range for predictions
x0_gd = np.linspace(x_min - x_padding, x_max + x_padding, 100).reshape(-1, 1)

# Create subplots
plt.figure(figsize=(18, 6))

# Loop through degrees and draw the polynomial regression plot
for i, degree in enumerate(degrees, start=1):
    plt.subplot(1, len(degrees), i)
    draw_polynomial_regression(X_train_scaled, Y_train, X_test_scaled, Y_test, x0_gd, degree, x_min, x_max, y_min, y_max, x_padding, y_padding)

# Adjust layout and display the plots
plt.tight_layout()
plt.show()
