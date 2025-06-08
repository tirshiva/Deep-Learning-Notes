# Linear Regression

## What is Linear Regression?

Linear regression is a statistical method that models the relationship between a dependent variable and one or more independent variables by fitting a linear equation to the observed data. It's one of the most fundamental and widely used techniques in both statistics and machine learning.

## Why is Linear Regression Important in Machine Learning?

1. **Foundation for Other Methods**
   - Basis for more complex models
   - Understanding model assumptions
   - Feature importance analysis

2. **Interpretability**
   - Clear relationship between variables
   - Easy to understand coefficients
   - Statistical significance testing

3. **Practical Applications**
   - Predictive modeling
   - Trend analysis
   - Feature selection

## How Does Linear Regression Work?

### 1. Simple Linear Regression

#### What is Simple Linear Regression?
**Definition**: Models relationship between one independent and one dependent variable
**Formula**: \[ y = \beta_0 + \beta_1x + \epsilon \]

```python
def simple_linear_regression(X, y):
    # Add intercept term
    X = np.column_stack([np.ones(len(X)), X])
    
    # Calculate coefficients using normal equation
    beta = np.linalg.inv(X.T @ X) @ X.T @ y
    
    return beta[0], beta[1]  # intercept, slope

# Example
X = np.array([1, 2, 3, 4, 5])
y = np.array([2, 4, 5, 4, 5])
intercept, slope = simple_linear_regression(X, y)
```

### 2. Multiple Linear Regression

#### What is Multiple Linear Regression?
**Definition**: Models relationship between multiple independent variables and one dependent variable
**Formula**: \[ y = \beta_0 + \beta_1x_1 + \beta_2x_2 + ... + \beta_nx_n + \epsilon \]

```python
def multiple_linear_regression(X, y):
    # Add intercept term
    X = np.column_stack([np.ones(len(X)), X])
    
    # Calculate coefficients using normal equation
    beta = np.linalg.inv(X.T @ X) @ X.T @ y
    
    return beta

# Example
X = np.random.randn(100, 3)
y = np.random.randn(100)
beta = multiple_linear_regression(X, y)
```

### 3. Model Evaluation

#### R-squared
**What**: Proportion of variance explained by the model
**Formula**: \[ R^2 = 1 - \frac{\sum(y_i - \hat{y}_i)^2}{\sum(y_i - \bar{y})^2} \]

```python
def calculate_r_squared(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)

# Example
y_true = np.array([1, 2, 3, 4, 5])
y_pred = np.array([1.1, 1.9, 3.1, 3.9, 5.1])
r_squared = calculate_r_squared(y_true, y_pred)
```

#### Mean Squared Error
**What**: Average squared difference between predictions and actual values
**Formula**: \[ MSE = \frac{1}{n}\sum(y_i - \hat{y}_i)^2 \]

```python
def calculate_mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

# Example
y_true = np.array([1, 2, 3, 4, 5])
y_pred = np.array([1.1, 1.9, 3.1, 3.9, 5.1])
mse = calculate_mse(y_true, y_pred)
```

## Visualizing Linear Regression

```python
import matplotlib.pyplot as plt
import seaborn as sns

def plot_regression_line(X, y, beta):
    plt.figure(figsize=(10, 6))
    plt.scatter(X, y, alpha=0.5, label='Data Points')
    
    # Plot regression line
    x_line = np.linspace(min(X), max(X), 100)
    y_line = beta[0] + beta[1] * x_line
    plt.plot(x_line, y_line, 'r-', label='Regression Line')
    
    plt.title('Linear Regression')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.legend()
    plt.show()

def plot_residuals(y_true, y_pred):
    plt.figure(figsize=(10, 6))
    residuals = y_true - y_pred
    
    plt.scatter(y_pred, residuals, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.title('Residual Plot')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.show()

# Example
X = np.random.normal(0, 1, 100)
y = 2 * X + np.random.normal(0, 0.5, 100)
beta = multiple_linear_regression(X.reshape(-1, 1), y)
plot_regression_line(X, y, beta)
```

## Common Pitfalls

1. **Multicollinearity**
   - Problem: Highly correlated features
   - Solution: Feature selection or regularization

2. **Non-linear Relationships**
   - Problem: Linear model may not fit
   - Solution: Feature transformation or non-linear models

3. **Outliers**
   - Problem: Can significantly affect results
   - Solution: Robust regression or outlier detection

## Applications in Machine Learning

### 1. Feature Importance
```python
def calculate_feature_importance(X, y):
    beta = multiple_linear_regression(X, y)
    return np.abs(beta[1:])  # Exclude intercept

# Example
X = np.random.randn(100, 3)
y = np.random.randn(100)
importance = calculate_feature_importance(X, y)
```

### 2. Prediction
```python
def predict(X, beta):
    X = np.column_stack([np.ones(len(X)), X])
    return X @ beta

# Example
X = np.random.randn(100, 3)
y = np.random.randn(100)
beta = multiple_linear_regression(X, y)
predictions = predict(X, beta)
```

### 3. Cross-validation
```python
def cross_validate(X, y, k=5):
    n_samples = len(X)
    indices = np.random.permutation(n_samples)
    fold_size = n_samples // k
    
    scores = []
    for i in range(k):
        test_indices = indices[i*fold_size:(i+1)*fold_size]
        train_indices = np.setdiff1d(indices, test_indices)
        
        X_train, X_test = X[train_indices], X[test_indices]
        y_train, y_test = y[train_indices], y[test_indices]
        
        beta = multiple_linear_regression(X_train, y_train)
        y_pred = predict(X_test, beta)
        score = calculate_r_squared(y_test, y_pred)
        scores.append(score)
    
    return np.mean(scores)

# Example
X = np.random.randn(100, 3)
y = np.random.randn(100)
cv_score = cross_validate(X, y)
```

## Interview Questions

1. **Basic Concepts**
   - What are the assumptions of linear regression?
   - How do you interpret regression coefficients?
   - What's the difference between R-squared and adjusted R-squared?

2. **Practical Applications**
   - How do you handle multicollinearity?
   - What's the role of regularization in linear regression?
   - How do you validate a linear regression model?

## Exercises

1. Implement different regression metrics:
   ```python
   # a) Root Mean Squared Error
   # b) Mean Absolute Error
   # c) Adjusted R-squared
   ```

2. Create a function for polynomial regression:
   ```python
   def polynomial_regression(X, y, degree=2):
       X_poly = np.column_stack([X**i for i in range(1, degree+1)])
       return multiple_linear_regression(X_poly, y)
   ```

3. Implement ridge regression:
   ```python
   def ridge_regression(X, y, alpha=1.0):
       X = np.column_stack([np.ones(len(X)), X])
       n_features = X.shape[1]
       return np.linalg.inv(X.T @ X + alpha * np.eye(n_features)) @ X.T @ y
   ```

## Additional Resources

- [Scikit-learn Linear Regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html)
- [StatsModels](https://www.statsmodels.org/stable/regression.html)
- [Linear Regression in Python](https://realpython.com/linear-regression-in-python/) 