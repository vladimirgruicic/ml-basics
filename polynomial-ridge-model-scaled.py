import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score

# Generate random data for demonstration
np.random.seed(100)
X = 2 * np.random.rand(100000, 1)
y = 4 * X**3 + 2 * np.random.rand(100000, 1)

# Introduce additional features
# Let's say you want to add a new features which is the square of the exisiting feature X
X_additional = X**4

# Concatenate the additional feature with original feature matrix.
X_combined = np.concatenate((X, X_additional), axis=1)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_combined, y, test_size=0.5, random_state=42)

# Generate polynomial features
poly = PolynomialFeatures(degree=1)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

# Define the regularization parameter
alpha = 50

# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_poly)
X_test_scaled = scaler.transform(X_test_poly)

# Create and train a Ridge regression model with scaled features
ridge_model_scaled = Ridge(alpha=alpha)
ridge_model_scaled.fit(X_train_scaled, y_train)

# Make predictions using the trained model
y_pred_test_ridge_scaled = ridge_model_scaled.predict(X_test_scaled)

# Evaluate ridge model performance on scaled test data
mse_test_ridge_scaled = mean_squared_error(y_test, y_pred_test_ridge_scaled)
r_squared_test_ridge_scaled = r2_score(y_test, y_pred_test_ridge_scaled)

# Cross-validation
cv_scores = cross_val_score(ridge_model_scaled, X_train_scaled, y_train, cv=5, scoring='r2')
print("Cross-Validation R-squared Scores:", cv_scores)
print("Mean Cross-Validation R-squared Score:", np.mean(cv_scores))

# Calculate baseline prediction
baseline_prediction = np.mean(y_train)
baseline_predictions = np.full_like(y_test, fill_value=baseline_prediction)

# Evaluate baseline performance
baseline_mse = mean_squared_error(y_test, baseline_predictions)
baseline_r_squared = r2_score(y_test, baseline_predictions)

# Compare with ridge model performance
print("\nBaseline Mean Squared Error (MSE):", baseline_mse)
print("Baseline R-squared:", baseline_r_squared)
print("\nRidge Regression with Scaled Features:")
print("Mean Squared Error (MSE) on Test Data:", mse_test_ridge_scaled)
print("R-squared on Test Data:", r_squared_test_ridge_scaled)

# Interpret the results
if mse_test_ridge_scaled < baseline_mse:
    print("Ridge model outperforms baseline.")
else:
    print("Baseline outperforms ridge model.")

# Visualize the results
plt.scatter(X_test[:,0], y_test, label='Test data')
plt.plot(X_test[:,0], y_pred_test_ridge_scaled, color='r', linestyle='-', linewidth=1, label='Predictions Ridge')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.title('Model Predictions on Test Data')
plt.show()
