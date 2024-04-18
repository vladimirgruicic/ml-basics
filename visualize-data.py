import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Generate random data for demonstration
np.random.seed(10)
X = 2 * np.random.rand(49, 1)
y = 4 + 3 * X + 5 * np.random.randn(49,1)

# Create and train a linear regression model
model = LinearRegression()
model.fit(X, y)

# Create an array of new data points for predictions
X_new = np.arange(0, 5, 0.1).reshape(-1, 1)

# Plot original data points
plt.scatter(X, y, label='Original data')

# Plot predictions made by the model
plt.plot(X_new, model.predict(X_new), 'r-', label='Predictions')

# Add labels and legend
plt.xlabel('X')
plt.ylabel('y')
plt.legend()

# Show the plot
plt.show()
