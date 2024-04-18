# import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Generate rnadom data for demonstration
np.random.seed(100)
X = 2 * np.random.rand(100, 1)
y = 4 * X + np.random.rand(100, 1)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Generate polynomial features
poly = PolynomialFeatures(degree=1)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

# Define the regularization parameter
alpha = 4.0

# Create and train a Ridge regression model
ridge_model = Ridge(alpha=alpha)
ridge_model.fit(X_train_poly, y_train)

# Make predictions using the trained model
y_pred_test_ridge = ridge_model.predict(X_test_poly)
                                        

# Evaluate ridge model performance on test data
mse_test_ridge = mean_squared_error(y_test, y_pred_test_ridge)
r_squared_test_ridge = r2_score(y_test, y_pred_test_ridge)

# Calculate baseline prediction
baseline_prediction = np.mean(y_train)
baseline_predictions = np.full_like(y_test, fill_value=baseline_prediction)

# Evaluate baseline performance
baseline_mse = mean_squared_error(y_test, baseline_predictions)
baseline_r_squared = r2_score(y_test, baseline_predictions)

# Compare with linear model performance
print("Baseline Mean Squared Error (MSE):", baseline_mse)
print("Baseline R-squared:", baseline_r_squared)

# Evaluate ridge model performance
print("\nRidge Regression:")
print("MEan Squared Error (MSE) on Test Data:", mse_test_ridge)
print("R-squared on Test Data:", r_squared_test_ridge)

# Interpret the results
if mse_test_ridge < baseline_mse:
    print("Ridge model outperforms baseline.")
else:
    print("Baseline outperforms ridge model.")

# Visualize the results
plt.scatter(X_test, y_test, label='Test data')
plt.plot(X_test, y_pred_test_ridge, color='r', linestyle='-', linewidth=1, label='Predictions Ridge')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.title('Model Predictions on Test Data')
plt.show()