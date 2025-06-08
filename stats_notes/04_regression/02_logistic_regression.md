# Logistic Regression

## What is Logistic Regression?

Logistic regression is a statistical method used for binary classification problems. Despite its name, it's a classification algorithm that uses a logistic function to model the probability of a binary outcome. It's one of the most widely used classification algorithms in machine learning.

## Why is Logistic Regression Important in Machine Learning?

1. **Binary Classification**
   - Predicting binary outcomes
   - Probability estimation
   - Decision boundary analysis

2. **Interpretability**
   - Clear probability outputs
   - Feature importance analysis
   - Odds ratio interpretation

3. **Foundation for Other Methods**
   - Basis for neural networks
   - Multi-class classification
   - Feature engineering

## How Does Logistic Regression Work?

### 1. The Logistic Function

#### What is the Logistic Function?
**Definition**: S-shaped function that maps any input to a value between 0 and 1
**Formula**: \[ \sigma(x) = \frac{1}{1 + e^{-x}} \]

```python
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Example
x = np.linspace(-10, 10, 100)
y = sigmoid(x)
```

### 2. Binary Logistic Regression

#### What is Binary Logistic Regression?
**Definition**: Models probability of binary outcome using logistic function
**Formula**: \[ P(Y=1|X) = \frac{1}{1 + e^{-(\beta_0 + \beta_1X)}} \]

```python
def logistic_regression(X, y, learning_rate=0.01, n_iterations=1000):
    n_samples, n_features = X.shape
    weights = np.zeros(n_features)
    bias = 0
    
    for _ in range(n_iterations):
        # Forward pass
        linear_pred = np.dot(X, weights) + bias
        predictions = sigmoid(linear_pred)
        
        # Compute gradients
        dw = (1/n_samples) * np.dot(X.T, (predictions - y))
        db = (1/n_samples) * np.sum(predictions - y)
        
        # Update parameters
        weights = weights - learning_rate * dw
        bias = bias - learning_rate * db
    
    return weights, bias

# Example
X = np.random.randn(100, 2)
y = np.random.randint(0, 2, 100)
weights, bias = logistic_regression(X, y)
```

### 3. Multi-class Logistic Regression

#### What is Multi-class Logistic Regression?
**Definition**: Extends binary logistic regression to multiple classes
**Formula**: \[ P(Y=k|X) = \frac{e^{\beta_kX}}{\sum_{j=1}^K e^{\beta_jX}} \]

```python
def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

def multiclass_logistic_regression(X, y, n_classes, learning_rate=0.01, n_iterations=1000):
    n_samples, n_features = X.shape
    weights = np.zeros((n_features, n_classes))
    bias = np.zeros(n_classes)
    
    # One-hot encode y
    y_onehot = np.zeros((n_samples, n_classes))
    y_onehot[np.arange(n_samples), y] = 1
    
    for _ in range(n_iterations):
        # Forward pass
        linear_pred = np.dot(X, weights) + bias
        predictions = softmax(linear_pred)
        
        # Compute gradients
        dw = (1/n_samples) * np.dot(X.T, (predictions - y_onehot))
        db = (1/n_samples) * np.sum(predictions - y_onehot, axis=0)
        
        # Update parameters
        weights = weights - learning_rate * dw
        bias = bias - learning_rate * db
    
    return weights, bias

# Example
X = np.random.randn(100, 2)
y = np.random.randint(0, 3, 100)  # 3 classes
weights, bias = multiclass_logistic_regression(X, y, n_classes=3)
```

## Model Evaluation

### 1. Classification Metrics

```python
def calculate_metrics(y_true, y_pred):
    # Confusion matrix
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    
    # Calculate metrics
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }

# Example
y_true = np.array([1, 0, 1, 1, 0])
y_pred = np.array([1, 0, 1, 0, 0])
metrics = calculate_metrics(y_true, y_pred)
```

### 2. ROC Curve and AUC

```python
def calculate_roc_auc(y_true, y_prob):
    thresholds = np.linspace(0, 1, 100)
    tpr = []  # True Positive Rate
    fpr = []  # False Positive Rate
    
    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)
        tp = np.sum((y_true == 1) & (y_pred == 1))
        tn = np.sum((y_true == 0) & (y_pred == 0))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        
        tpr.append(tp / (tp + fn) if (tp + fn) > 0 else 0)
        fpr.append(fp / (fp + tn) if (fp + tn) > 0 else 0)
    
    # Calculate AUC using trapezoidal rule
    auc = np.trapz(tpr, fpr)
    
    return fpr, tpr, auc

# Example
y_true = np.array([1, 0, 1, 1, 0])
y_prob = np.array([0.8, 0.2, 0.7, 0.6, 0.3])
fpr, tpr, auc = calculate_roc_auc(y_true, y_prob)
```

## Visualizing Logistic Regression

```python
import matplotlib.pyplot as plt
import seaborn as sns

def plot_decision_boundary(X, y, weights, bias):
    plt.figure(figsize=(10, 6))
    
    # Plot data points
    plt.scatter(X[y==0, 0], X[y==0, 1], label='Class 0')
    plt.scatter(X[y==1, 0], X[y==1, 1], label='Class 1')
    
    # Plot decision boundary
    x1 = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)
    x2 = -(weights[0] * x1 + bias) / weights[1]
    plt.plot(x1, x2, 'r-', label='Decision Boundary')
    
    plt.title('Logistic Regression Decision Boundary')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.show()

def plot_roc_curve(fpr, tpr, auc):
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.show()

# Example
X = np.random.randn(100, 2)
y = np.random.randint(0, 2, 100)
weights, bias = logistic_regression(X, y)
plot_decision_boundary(X, y, weights, bias)
```

## Common Pitfalls

1. **Class Imbalance**
   - Problem: Uneven class distribution
   - Solution: Resampling or class weights

2. **Overfitting**
   - Problem: Model too complex
   - Solution: Regularization

3. **Feature Scaling**
   - Problem: Features on different scales
   - Solution: Standardization or normalization

## Applications in Machine Learning

### 1. Binary Classification
```python
def predict_binary(X, weights, bias, threshold=0.5):
    linear_pred = np.dot(X, weights) + bias
    probabilities = sigmoid(linear_pred)
    return (probabilities >= threshold).astype(int)

# Example
X = np.random.randn(100, 2)
weights, bias = logistic_regression(X, y)
predictions = predict_binary(X, weights, bias)
```

### 2. Probability Estimation
```python
def predict_probability(X, weights, bias):
    linear_pred = np.dot(X, weights) + bias
    return sigmoid(linear_pred)

# Example
X = np.random.randn(100, 2)
weights, bias = logistic_regression(X, y)
probabilities = predict_probability(X, weights, bias)
```

### 3. Feature Importance
```python
def calculate_feature_importance(weights):
    return np.abs(weights)

# Example
X = np.random.randn(100, 2)
y = np.random.randint(0, 2, 100)
weights, bias = logistic_regression(X, y)
importance = calculate_feature_importance(weights)
```

## Interview Questions

1. **Basic Concepts**
   - What's the difference between linear and logistic regression?
   - How do you interpret logistic regression coefficients?
   - What's the role of the sigmoid function?

2. **Practical Applications**
   - How do you handle multi-class problems?
   - What's the role of regularization in logistic regression?
   - How do you choose the decision threshold?

## Exercises

1. Implement different regularization methods:
   ```python
   # a) L1 regularization (Lasso)
   # b) L2 regularization (Ridge)
   # c) Elastic Net
   ```

2. Create a function for cross-validation:
   ```python
   def cross_validate_logistic(X, y, k=5):
       n_samples = len(X)
       indices = np.random.permutation(n_samples)
       fold_size = n_samples // k
       
       scores = []
       for i in range(k):
           test_indices = indices[i*fold_size:(i+1)*fold_size]
           train_indices = np.setdiff1d(indices, test_indices)
           
           X_train, X_test = X[train_indices], X[test_indices]
           y_train, y_test = y[train_indices], y[test_indices]
           
           weights, bias = logistic_regression(X_train, y_train)
           y_pred = predict_binary(X_test, weights, bias)
           score = calculate_metrics(y_test, y_pred)['accuracy']
           scores.append(score)
       
       return np.mean(scores)
   ```

3. Implement a function for handling class imbalance:
   ```python
   def balanced_logistic_regression(X, y, class_weights=None):
       if class_weights is None:
           class_weights = {
               0: len(y) / (2 * np.sum(y == 0)),
               1: len(y) / (2 * np.sum(y == 1))
           }
       return logistic_regression(X, y, class_weights=class_weights)
   ```

## Additional Resources

- [Scikit-learn Logistic Regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
- [StatsModels](https://www.statsmodels.org/stable/discretemod.html)
- [Logistic Regression in Python](https://realpython.com/logistic-regression-python/) 