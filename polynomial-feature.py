# import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Generate rnadom data for demonstration
np.random.seed(40)
X = 2 * np.random.rand(300, 1)
y = 4 * X + np.random.rand(300, 1)

# Generate polynomial features
poly = PolynomialFeatures(degree=6)
X_poly = poly.fit_transform(X)

# Create and train a linear regression model
model = LinearRegression()
model.fit(X_poly, y)

# Generate new data points for prediction
X_new = np.linspace(0, 2, 100).reshape(-1, 1)

# Transform the new data points using polynomial features
X_new_poly = poly.transform(X_new)

# Make predictions using the trained model
y_pred_poly = model.predict(X_new_poly)

# Visualize the results.
plt.scatter(X, y, label='Original data')
plt.plot(X_new, y_pred_poly, 'r-', label='Predictions with polynomial features')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.show()