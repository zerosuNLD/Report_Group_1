import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge

# Feature scaling
def feature_scaling(X):
    return (X - np.mean(X)) / np.std(X)

# Combine polynomial features
def capacity(X, degree):
    m = X.shape[0]
    X_combined = np.ones((m, 1))  # Start with a column of ones for the bias term
    for d in range(1, degree + 1):
        X_combined = np.hstack((X_combined, X**d))
    return X_combined

# Analytical solution using the normal equation
def formular(X_combined, Y):
    return np.linalg.inv(X_combined.T.dot(X_combined)).dot(X_combined.T).dot(Y)

# Data
X = np.array([[33, 45, 49, 43, 51, 59, 67, 63, 69, 58, 34, 32, 20, 22, 26, 76, 80, 62, 55, 48, 44, 92, 100, 105, 110, 86, 87, 114, 103, 96, 131, 140, 139, 134, 162, 132, 150, 153, 162, 117, 125, 175, 184, 174, 174, 151, 146, 190, 183, 200, 191, 191, 211, 213, 202, 203, 216, 204, 221, 237, 215, 233, 232, 253, 242, 239, 250, 251, 260, 263, 256, 240, 226, 229, 237, 243, 251, 294, 271, 268, 281, 289, 266, 253]]).T
Y = np.array([[17, 33, 26, 21, 46, 37, 57, 52, 46, 46, 26, 7, 15, 7, 23, 67, 60, 70, 60, 53, 48, 58, 46, 39, 33, 61, 52, 24, 29, 39, 37, 47, 26, 20, 48, 30, 50, 40, 69, 15, 15, 67, 83, 94, 86, 75, 67, 60, 56, 50, 52, 76, 72, 63, 69, 81, 52, 37, 34, 64, 28, 50, 26, 33, 26, 16, 0, 17, 0, -13, -17, 0, 17, 5, -10, -25, -36, -57, -44, -55, -63, -35, -28, -64]]).T

# Split data into training and testing sets (70/30)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

# Parameters
feature_count = 25
optimal_feature_count = 5
alpha = 0.1  # Regularization strength for L2

# Feature scaling
X_train_scaled = feature_scaling(X_train)
X_test_scaled = feature_scaling(X_test)

# Combine features for polynomial regression of degree feature_count and optimal_feature_count
X_combined_train = capacity(X_train_scaled, feature_count)
X_combined_test = capacity(X_test_scaled, feature_count)
X_combined_optimal_train = capacity(X_train_scaled, optimal_feature_count)
X_combined_optimal_test = capacity(X_test_scaled, optimal_feature_count)

# Calculate theta values without regularization
theta_default = formular(X_combined_train, Y_train)

# Train Ridge regression model (L2 regularization)
ridge_model = Ridge(alpha=alpha, max_iter=10000)
ridge_model.fit(X_combined_train, Y_train.ravel())

# Calculate predicted Y values
Y_predict_default_train = X_combined_train.dot(theta_default)
Y_predict_default_test = X_combined_test.dot(theta_default)
Y_predict_ridge_train = ridge_model.predict(X_combined_train)
Y_predict_ridge_test = ridge_model.predict(X_combined_test)

# Tính toán lỗi
mse_default_train = mean_squared_error(Y_train, Y_predict_default_train)
mse_default_test = mean_squared_error(Y_test, Y_predict_default_test)
mse_ridge_train = mean_squared_error(Y_train, Y_predict_ridge_train)
mse_ridge_test = mean_squared_error(Y_test, Y_predict_ridge_test)

print(f"Linear Regression MSE - Train: {mse_default_train}, Test: {mse_default_test}")
print(f"Ridge Regression MSE - Train: {mse_ridge_train}, Test: {mse_ridge_test}")

# Calculate axis limits based on the data
x_min, x_max = X_train_scaled.min(), X_train_scaled.max()
y_min, y_max = Y_train.min(), Y_train.max()

# Add padding around min and max values to avoid edge clipping
x_padding = (x_max - x_min) * 0.1
y_padding = (y_max - y_min) * 0.1

# Generate x0 range for predictions
x0 = np.linspace(x_min - x_padding, x_max + x_padding, 100).reshape(-1, 1)

# Combine features for polynomial regression of degree feature_count and optimal_feature_count
x0_combined = capacity(x0, feature_count)
x0_combined_optimal = capacity(x0, optimal_feature_count)

# Calculate predicted Y values for plotting
Y_predict_default_plot = x0_combined.dot(theta_default)
Y_predict_ridge_plot = ridge_model.predict(x0_combined)
Y_predict_optimal_plot = x0_combined_optimal.dot(formular(X_combined_optimal_train, Y_train))

# Prepare figure for plotting
fig, axs = plt.subplots(1, 2, figsize=(16, 6))

# Plot for Linear Regression and Optimal Capacity
axs[0].plot(X_train_scaled, Y_train, 'ro', label='Training Data')
axs[0].plot(X_test_scaled, Y_test, 'bo', label='Test Data')
axs[0].plot(x0, Y_predict_default_plot, label='Linear Regression')
axs[0].plot(x0, Y_predict_optimal_plot, label='Optimal Capacity', linestyle='--')
axs[0].set_xlim(x_min - x_padding, x_max + x_padding)
axs[0].set_ylim(y_min - y_padding, y_max + y_padding)
axs[0].set_xlabel('X (scaled)')
axs[0].set_ylabel('Y')
axs[0].set_title('Linear Regression')
axs[0].legend(loc='upper right')
axs[0].grid(True)

# Add MSE values to the plot
textstr_default = f'Linear Regression MSE:\nTrain: {mse_default_train:.2f}\nTest: {mse_default_test:.2f}'
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
axs[0].text(0.05, 0.95, textstr_default, transform=axs[0].transAxes, fontsize=10,
            verticalalignment='top', bbox=props)

# Plot for Ridge Regression and Optimal Capacity
axs[1].plot(X_train_scaled, Y_train, 'ro', label='Training Data')
axs[1].plot(X_test_scaled, Y_test, 'bo', label='Test Data')
axs[1].plot(x0, Y_predict_ridge_plot, label='Ridge Regression')
axs[1].plot(x0, Y_predict_optimal_plot, label='Optimal Capacity', linestyle='--')
axs[1].set_xlim(x_min - x_padding, x_max + x_padding)
axs[1].set_ylim(y_min - y_padding, y_max + y_padding)
axs[1].set_xlabel('X (scaled)')
axs[1].set_ylabel('Y')
axs[1].set_title('Ridge Regression')
axs[1].legend(loc='upper right')
axs[1].grid(True)

# Add MSE values to the plot
textstr_ridge = f'Ridge Regression MSE:\nTrain: {mse_ridge_train:.2f}\nTest: {mse_ridge_test:.2f}'
axs[1].text(0.05, 0.95, textstr_ridge, transform=axs[1].transAxes, fontsize=10,
            verticalalignment='top', bbox=props)

plt.tight_layout()
plt.show()
