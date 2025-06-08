# Linear Regression

## Background and Introduction
Linear regression is one of the most fundamental and widely used algorithms in machine learning. It was developed by Francis Galton in the 19th century and has since become a cornerstone of statistical modeling and machine learning.

## What is Linear Regression?
Linear regression is a statistical method that models the relationship between a dependent variable (target) and one or more independent variables (features) using a linear equation. The goal is to find the best-fitting line that minimizes the sum of squared errors between predicted and actual values.

## Why Linear Regression?
1. **Simplicity**: Easy to understand and implement
2. **Interpretability**: Results are easily interpretable
3. **Baseline Model**: Often used as a baseline for comparison
4. **Feature Importance**: Helps understand feature relationships
5. **Wide Applications**: Used in various fields (economics, science, etc.)

## How Does Linear Regression Work?

### 1. Simple Linear Regression
For a single feature, the model is:
\[ y = \beta_0 + \beta_1x + \epsilon \]
where:
- \(y\) is the target variable
- \(x\) is the feature
- \(\beta_0\) is the y-intercept
- \(\beta_1\) is the slope
- \(\epsilon\) is the error term

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Generate sample data
np.random.seed(42)
X = np.random.rand(100, 1)
y = 2 * X + 1 + 0.1 * np.random.randn(100, 1)

# Create and train model
model = LinearRegression()
model.fit(X, y)

# Make predictions
y_pred = model.predict(X)

# Calculate metrics
mse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)

# Plot results
plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='blue', label='Data')
plt.plot(X, y_pred, color='red', label='Regression Line')
plt.title('Simple Linear Regression')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.show()

print(f"Slope: {model.coef_[0][0]:.2f}")
print(f"Intercept: {model.intercept_[0]:.2f}")
print(f"MSE: {mse:.4f}")
print(f"R²: {r2:.4f}")
```

### 2. Multiple Linear Regression
For multiple features, the model is:
\[ y = \beta_0 + \beta_1x_1 + \beta_2x_2 + ... + \beta_nx_n + \epsilon \]

```python
# Generate sample data
np.random.seed(42)
X = np.random.rand(100, 2)
y = 2 * X[:, 0] + 3 * X[:, 1] + 1 + 0.1 * np.random.randn(100)

# Create and train model
model = LinearRegression()
model.fit(X, y)

# Make predictions
y_pred = model.predict(X)

# Calculate metrics
mse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)

print(f"Coefficients: {model.coef_}")
print(f"Intercept: {model.intercept_:.2f}")
print(f"MSE: {mse:.4f}")
print(f"R²: {r2:.4f}")
```

### 3. Implementation from Scratch
```python
class LinearRegressionScratch:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        
    def fit(self, X, y):
        n_samples, n_features = X.shape
        
        # Initialize parameters
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        # Gradient descent
        for _ in range(self.n_iterations):
            # Forward pass
            y_predicted = np.dot(X, self.weights) + self.bias
            
            # Compute gradients
            dw = (1/n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1/n_samples) * np.sum(y_predicted - y)
            
            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
    def predict(self, X):
        return np.dot(X, self.weights) + self.bias
```

## Model Evaluation

### 1. Metrics
```python
def evaluate_model(y_true, y_pred):
    # Mean Squared Error
    mse = mean_squared_error(y_true, y_pred)
    
    # Root Mean Squared Error
    rmse = np.sqrt(mse)
    
    # R-squared
    r2 = r2_score(y_true, y_pred)
    
    # Adjusted R-squared
    n = len(y_true)
    p = X.shape[1]
    adj_r2 = 1 - (1-r2)*(n-1)/(n-p-1)
    
    return {
        'MSE': mse,
        'RMSE': rmse,
        'R²': r2,
        'Adjusted R²': adj_r2
    }
```

### 2. Assumptions
1. **Linearity**: Relationship between features and target is linear
2. **Independence**: Observations are independent
3. **Homoscedasticity**: Constant variance of errors
4. **Normality**: Errors are normally distributed

```python
def check_assumptions(X, y, y_pred):
    # Calculate residuals
    residuals = y - y_pred
    
    # Plot residuals
    plt.figure(figsize=(12, 4))
    
    # Residuals vs Predicted
    plt.subplot(1, 2, 1)
    plt.scatter(y_pred, residuals)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.title('Residuals vs Predicted')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    
    # Residuals Histogram
    plt.subplot(1, 2, 2)
    plt.hist(residuals, bins=30)
    plt.title('Residuals Distribution')
    plt.xlabel('Residuals')
    plt.ylabel('Frequency')
    
    plt.tight_layout()
    plt.show()
```

## Common Interview Questions

1. **Q: What is the difference between simple and multiple linear regression?**
   - A: Simple linear regression uses one feature to predict the target, while multiple linear regression uses multiple features. The mathematical form and interpretation differ, but the core concept remains the same.

2. **Q: How do you handle multicollinearity in linear regression?**
   - A: Common approaches include:
     - Removing correlated features
     - Using regularization (Ridge, Lasso)
     - Principal Component Analysis
     - Feature selection techniques

3. **Q: What are the advantages and disadvantages of linear regression?**
   - A: Advantages:
     - Simple and interpretable
     - Fast to train and predict
     - Works well with linear relationships
     Disadvantages:
     - Assumes linear relationship
     - Sensitive to outliers
     - May underfit complex patterns

## Hands-on Task: House Price Prediction

### Project: Real Estate Price Prediction
```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Create sample housing data
np.random.seed(42)
n_houses = 1000
data = {
    'area': np.random.normal(2000, 500, n_houses),
    'bedrooms': np.random.randint(1, 6, n_houses),
    'bathrooms': np.random.randint(1, 4, n_houses),
    'age': np.random.randint(0, 50, n_houses),
    'price': np.random.normal(300000, 100000, n_houses)
}
df = pd.DataFrame(data)

# Prepare features and target
X = df.drop('price', axis=1)
y = df['price']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = model.predict(X_test_scaled)

# Evaluate model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print results
print("Model Coefficients:")
for feature, coef in zip(X.columns, model.coef_):
    print(f"{feature}: {coef:.2f}")
print(f"\nIntercept: {model.intercept_:.2f}")
print(f"MSE: {mse:.2f}")
print(f"R²: {r2:.4f}")

# Visualize predictions
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.title('Actual vs Predicted Prices')
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.show()
```

## Next Steps
1. Learn about regularization techniques
2. Study polynomial regression
3. Explore advanced regression models
4. Practice with real-world datasets
5. Learn about model interpretation

## Resources
- [Scikit-learn Linear Regression](https://scikit-learn.org/stable/modules/linear_model.html)
- [Introduction to Linear Regression](https://www.coursera.org/learn/linear-regression)
- [Linear Regression in Python](https://www.kaggle.com/learn/linear-regression)
- [Statistics for Machine Learning](https://www.amazon.com/Statistics-Machine-Learning-Techniques-Understanding/dp/1484233475) 