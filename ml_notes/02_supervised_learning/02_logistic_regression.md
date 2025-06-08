# Logistic Regression

## Background and Introduction
Logistic regression is a fundamental classification algorithm that models the probability of a binary outcome. Despite its name, it's a classification algorithm, not a regression algorithm. It was developed in the 1950s and remains one of the most widely used classification methods.

## What is Logistic Regression?
Logistic regression is a statistical method that uses a logistic function (sigmoid) to model binary outcomes. It estimates the probability that an instance belongs to a particular class, making it suitable for binary classification problems.

## Why Logistic Regression?
1. **Interpretability**: Results are easily interpretable
2. **Probability Output**: Provides probability estimates
3. **Efficiency**: Fast to train and predict
4. **Baseline Model**: Often used as a baseline for classification
5. **Feature Importance**: Helps understand feature relationships

## How Does Logistic Regression Work?

### 1. The Logistic Function
The sigmoid function transforms linear output into probabilities:
\[ \sigma(z) = \frac{1}{1 + e^{-z}} \]
where \(z = \beta_0 + \beta_1x_1 + \beta_2x_2 + ... + \beta_nx_n\)

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Plot sigmoid function
z = np.linspace(-10, 10, 100)
plt.figure(figsize=(10, 6))
plt.plot(z, sigmoid(z))
plt.title('Sigmoid Function')
plt.xlabel('z')
plt.ylabel('σ(z)')
plt.grid(True)
plt.show()
```

### 2. Binary Classification
```python
# Generate sample data
np.random.seed(42)
X = np.random.randn(100, 2)
y = (X[:, 0] + X[:, 1] > 0).astype(int)

# Create and train model
model = LogisticRegression()
model.fit(X, y)

# Make predictions
y_pred = model.predict(X)
y_prob = model.predict_proba(X)

# Calculate accuracy
accuracy = accuracy_score(y, y_pred)

# Print classification report
print(classification_report(y, y_pred))

# Plot decision boundary
def plot_decision_boundary(X, y, model):
    h = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                        np.arange(y_min, y_max, h))
    
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    plt.figure(figsize=(10, 6))
    plt.contourf(xx, yy, Z, alpha=0.4)
    plt.scatter(X[:, 0], X[:, 1], c=y, alpha=0.8)
    plt.title('Decision Boundary')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.show()

plot_decision_boundary(X, y, model)
```

### 3. Implementation from Scratch
```python
class LogisticRegressionScratch:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        
        # Initialize parameters
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        # Gradient descent
        for _ in range(self.n_iterations):
            # Forward pass
            linear_pred = np.dot(X, self.weights) + self.bias
            predictions = self.sigmoid(linear_pred)
            
            # Compute gradients
            dw = (1/n_samples) * np.dot(X.T, (predictions - y))
            db = (1/n_samples) * np.sum(predictions - y)
            
            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
    def predict(self, X):
        linear_pred = np.dot(X, self.weights) + self.bias
        y_pred = self.sigmoid(linear_pred)
        return (y_pred > 0.5).astype(int)
    
    def predict_proba(self, X):
        linear_pred = np.dot(X, self.weights) + self.bias
        return self.sigmoid(linear_pred)
```

## Model Evaluation

### 1. Metrics
```python
from sklearn.metrics import confusion_matrix, roc_curve, auc

def evaluate_classifier(y_true, y_pred, y_prob):
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    
    # Plot ROC Curve
    plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.show()
    
    return {
        'Confusion Matrix': cm,
        'AUC': roc_auc
    }
```

### 2. Regularization
```python
# L1 Regularization (Lasso)
model_l1 = LogisticRegression(penalty='l1', solver='liblinear')
model_l1.fit(X, y)

# L2 Regularization (Ridge)
model_l2 = LogisticRegression(penalty='l2')
model_l2.fit(X, y)

# Elastic Net
model_elastic = LogisticRegression(penalty='elasticnet', 
                                 l1_ratio=0.5,
                                 solver='saga')
model_elastic.fit(X, y)
```

## Common Interview Questions

1. **Q: What is the difference between linear and logistic regression?**
   - A: Linear regression predicts continuous values, while logistic regression predicts probabilities for binary outcomes. Logistic regression uses the sigmoid function to transform linear output into probabilities.

2. **Q: How do you handle multiclass classification with logistic regression?**
   - A: Common approaches include:
     - One-vs-Rest (OvR)
     - One-vs-One (OvO)
     - Multinomial logistic regression
     - Softmax regression

3. **Q: What are the advantages and disadvantages of logistic regression?**
   - A: Advantages:
     - Simple and interpretable
     - Provides probability estimates
     - Works well with linear decision boundaries
     Disadvantages:
     - Assumes linear decision boundary
     - May underfit complex patterns
     - Sensitive to outliers

## Hands-on Task: Customer Churn Prediction

### Project: Telecom Customer Churn Analysis
```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score

# Create sample customer data
np.random.seed(42)
n_customers = 1000
data = {
    'tenure': np.random.normal(24, 12, n_customers),
    'monthly_charges': np.random.normal(50, 20, n_customers),
    'contract_type': np.random.choice(['monthly', 'yearly', 'two-year'], n_customers),
    'payment_method': np.random.choice(['credit', 'debit', 'bank transfer'], n_customers),
    'churn': np.random.randint(0, 2, n_customers)
}
df = pd.DataFrame(data)

# Prepare features and target
X = pd.get_dummies(df.drop('churn', axis=1))
y = df['churn']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)[:, 1]

# Evaluate model
print("Classification Report:")
print(classification_report(y_test, y_pred))
print(f"\nROC AUC Score: {roc_auc_score(y_test, y_prob):.4f}")

# Feature importance
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': model.coef_[0]
})
feature_importance = feature_importance.sort_values('Coefficient', ascending=False)
print("\nFeature Importance:")
print(feature_importance)

# Visualize feature importance
plt.figure(figsize=(10, 6))
plt.barh(feature_importance['Feature'], feature_importance['Coefficient'])
plt.title('Feature Importance')
plt.xlabel('Coefficient')
plt.show()
```

## Next Steps
1. Learn about regularization techniques
2. Study multiclass classification
3. Explore advanced classification models
4. Practice with real-world datasets
5. Learn about model interpretation

## Resources
- [Scikit-learn Logistic Regression](https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression)
- [Introduction to Logistic Regression](https://www.coursera.org/learn/logistic-regression)
- [Logistic Regression in Python](https://www.kaggle.com/learn/logistic-regression)
- [Classification and Regression Trees](https://www.amazon.com/Classification-Regression-Trees-Leo-Breiman/dp/0412048418) 