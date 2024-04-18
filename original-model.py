# import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Generate random data for demonstration
np.random.seed(10)
X = 2 * np.random.rand(49, 1)
y = 4 + 3 * X + 5 * np.random.randn(49,1)

# Print the first 5 elements of X and y
print("First 5 elements of X:")
print(X[:5])
print("\nFirst 5 elements of y:")
print(y[:5])

# Create and train a linear regression model
model = LinearRegression()
model.fit(X, y)

# Check the model paramters
print("\nIntercept", model.intercept_)
print("Slope", model.coef_)

# Make predictions using the trained model
X_new = np.array([[0], [2]])
y_pred = model.predict(X_new)

# Check the predictions.
print("\nPredicted values for X_new:")
print(y_pred)

# Visualize the results.
plt.scatter(X, y)
plt.plot(X_new, y_pred, 'r-', label='Linear regression')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.show()