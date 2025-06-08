# Support Vector Machines (SVM)

## Background and Introduction
Support Vector Machines (SVM) is a powerful supervised learning algorithm developed in the 1990s. It's particularly effective in high-dimensional spaces and is widely used for classification and regression tasks. The key idea behind SVM is to find the optimal hyperplane that maximizes the margin between classes.

## What is SVM?
SVM is a discriminative classifier that finds the optimal hyperplane to separate classes in the feature space. The algorithm aims to maximize the margin between the decision boundary and the nearest data points (support vectors) from each class.

## Why SVM?
1. **Effective in High Dimensions**: Works well even when the number of features is greater than the number of samples
2. **Memory Efficient**: Uses only support vectors for prediction
3. **Versatile**: Can handle both linear and non-linear classification
4. **Robust**: Less prone to overfitting in high-dimensional spaces
5. **Kernel Trick**: Can solve non-linear problems by mapping data to higher dimensions

## How Does SVM Work?

### 1. Linear SVM
```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler

# Generate sample data
X, y = make_blobs(n_samples=100, centers=2, random_state=42)

# Create and train SVM
svm = SVC(kernel='linear')
svm.fit(X, y)

# Plot decision boundary
def plot_decision_boundary(X, y, model, title):
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
    plt.title(title)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.show()

plot_decision_boundary(X, y, svm, 'Linear SVM Decision Boundary')
```

### 2. Kernel SVM
```python
# Generate non-linear data
X, y = make_blobs(n_samples=100, centers=2, random_state=42)
X = X @ np.array([[0.5, 0.5], [0.5, -0.5]])

# Create and train different kernel SVMs
kernels = ['linear', 'poly', 'rbf', 'sigmoid']
plt.figure(figsize=(15, 10))

for i, kernel in enumerate(kernels, 1):
    svm = SVC(kernel=kernel)
    svm.fit(X, y)
    
    plt.subplot(2, 2, i)
    plot_decision_boundary(X, y, svm, f'{kernel.capitalize()} Kernel SVM')
plt.tight_layout()
plt.show()
```

### 3. SVM Implementation from Scratch
```python
class SVM:
    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iters=1000):
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.w = None
        self.b = None
        
    def fit(self, X, y):
        n_samples, n_features = X.shape
        
        # Convert labels to -1 and 1
        y_ = np.where(y <= 0, -1, 1)
        
        # Initialize parameters
        self.w = np.zeros(n_features)
        self.b = 0
        
        # Gradient descent
        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                condition = y_[idx] * (np.dot(x_i, self.w) - self.b) >= 1
                
                if condition:
                    self.w -= self.lr * (2 * self.lambda_param * self.w)
                else:
                    self.w -= self.lr * (2 * self.lambda_param * self.w - 
                                       np.dot(x_i, y_[idx]))
                    self.b -= self.lr * y_[idx]
    
    def predict(self, X):
        return np.sign(np.dot(X, self.w) - self.b)
```

## Model Evaluation

### 1. Cross-validation and Hyperparameter Tuning
```python
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report

# Define parameter grid
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto', 0.1, 0.01],
    'kernel': ['rbf', 'linear', 'poly']
}

# Create and train model with grid search
svm = SVC()
grid_search = GridSearchCV(svm, param_grid, cv=5)
grid_search.fit(X, y)

# Print best parameters
print("Best parameters:", grid_search.best_params_)

# Evaluate model
y_pred = grid_search.predict(X)
print("\nClassification Report:")
print(classification_report(y, y_pred))
```

### 2. Support Vectors Visualization
```python
def plot_support_vectors(X, y, model):
    plt.figure(figsize=(10, 6))
    plt.scatter(X[:, 0], X[:, 1], c=y, alpha=0.8)
    
    # Plot support vectors
    if hasattr(model, 'support_vectors_'):
        plt.scatter(model.support_vectors_[:, 0],
                   model.support_vectors_[:, 1],
                   s=100, linewidth=1, facecolors='none',
                   edgecolors='k', label='Support Vectors')
    
    plt.title('Support Vectors')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.show()

# Train and plot
svm = SVC(kernel='rbf')
svm.fit(X, y)
plot_support_vectors(X, y, svm)
```

## Common Interview Questions

1. **Q: What is the kernel trick in SVM?**
   - A: The kernel trick allows SVM to solve non-linear classification problems by implicitly mapping the input data into higher-dimensional feature spaces. Common kernels include:
     - Linear: \(K(x,y) = x^T y\)
     - Polynomial: \(K(x,y) = (γx^T y + r)^d\)
     - RBF: \(K(x,y) = exp(-γ||x-y||^2)\)
     - Sigmoid: \(K(x,y) = tanh(γx^T y + r)\)

2. **Q: How do you handle multi-class classification with SVM?**
   - A: Common approaches include:
     - One-vs-Rest (OvR)
     - One-vs-One (OvO)
     - Directed Acyclic Graph SVM (DAGSVM)
     - Error-Correcting Output Codes (ECOC)

3. **Q: What are the advantages and disadvantages of SVM?**
   - A: Advantages:
     - Effective in high-dimensional spaces
     - Memory efficient
     - Versatile through kernel trick
     - Robust against overfitting
     Disadvantages:
     - Sensitive to parameter tuning
     - Computationally expensive for large datasets
     - Requires feature scaling
     - Black box model (less interpretable)

## Hands-on Task: Image Classification

### Project: Digit Recognition with SVM
```python
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Load digits dataset
digits = load_digits()
X = digits.data
y = digits.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train SVM
svm = SVC(kernel='rbf', C=10, gamma='scale')
svm.fit(X_train_scaled, y_train)

# Make predictions
y_pred = svm.predict(X_test_scaled)

# Evaluate model
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Plot confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Visualize some predictions
def plot_predictions(X_test, y_test, y_pred, indices):
    plt.figure(figsize=(15, 5))
    for i, idx in enumerate(indices):
        plt.subplot(1, len(indices), i + 1)
        plt.imshow(X_test[idx].reshape(8, 8), cmap='gray')
        plt.title(f'True: {y_test[idx]}\nPred: {y_pred[idx]}')
        plt.axis('off')
    plt.show()

# Plot some example predictions
plot_predictions(X_test, y_test, y_pred, indices=range(5))
```

## Next Steps
1. Learn about different kernel functions
2. Study advanced SVM variants (ν-SVM, One-class SVM)
3. Explore SVM for regression (SVR)
4. Practice with real-world datasets
5. Learn about model interpretation techniques

## Resources
- [Scikit-learn SVM](https://scikit-learn.org/stable/modules/svm.html)
- [Introduction to SVM](https://www.coursera.org/learn/support-vector-machines)
- [SVM in Python](https://www.kaggle.com/learn/support-vector-machines)
- [Support Vector Machines](https://www.amazon.com/Support-Vector-Machines-Olivier-Chapelle/dp/0262033949) 