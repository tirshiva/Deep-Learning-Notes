# K-Nearest Neighbors (KNN)

## Background and Introduction
K-Nearest Neighbors (KNN) is one of the simplest and most intuitive machine learning algorithms. It's an instance-based learning algorithm that makes predictions based on the similarity of new instances to known instances in the training data. The algorithm was first introduced in the 1950s and remains popular due to its simplicity and effectiveness.

## What is KNN?
KNN is a non-parametric, lazy learning algorithm that classifies new instances based on the majority class of their K nearest neighbors in the feature space. The algorithm:
1. Stores all training instances
2. Computes distances between new instance and all training instances
3. Selects K nearest neighbors
4. Makes prediction based on majority class (classification) or average value (regression)

## Why KNN?
1. **Simple to Understand**: Easy to implement and interpret
2. **No Training Phase**: Makes predictions directly from training data
3. **Adaptable**: Works for both classification and regression
4. **Non-linear**: Can capture complex decision boundaries
5. **No Assumptions**: Makes no assumptions about data distribution

## How Does KNN Work?

### 1. Basic Implementation
```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Generate sample data
X, y = make_blobs(n_samples=100, centers=2, random_state=42)

# Create and train KNN
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X, y)

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

plot_decision_boundary(X, y, knn, 'KNN Decision Boundary')
```

### 2. KNN Implementation from Scratch
```python
class KNN:
    def __init__(self, k=5, distance_metric='euclidean'):
        self.k = k
        self.distance_metric = distance_metric
        self.X_train = None
        self.y_train = None
        
    def _euclidean_distance(self, x1, x2):
        return np.sqrt(np.sum((x1 - x2) ** 2))
    
    def _manhattan_distance(self, x1, x2):
        return np.sum(np.abs(x1 - x2))
    
    def _minkowski_distance(self, x1, x2, p=3):
        return np.power(np.sum(np.power(np.abs(x1 - x2), p)), 1/p)
    
    def _compute_distance(self, x1, x2):
        if self.distance_metric == 'euclidean':
            return self._euclidean_distance(x1, x2)
        elif self.distance_metric == 'manhattan':
            return self._manhattan_distance(x1, x2)
        elif self.distance_metric == 'minkowski':
            return self._minkowski_distance(x1, x2)
        else:
            raise ValueError("Distance metric not supported")
    
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
    
    def predict(self, X):
        y_pred = []
        
        for x in X:
            # Compute distances
            distances = [self._compute_distance(x, x_train) 
                        for x_train in self.X_train]
            
            # Get k nearest neighbors
            k_indices = np.argsort(distances)[:self.k]
            k_nearest_labels = [self.y_train[i] for i in k_indices]
            
            # Majority vote
            most_common = max(set(k_nearest_labels), 
                            key=k_nearest_labels.count)
            y_pred.append(most_common)
        
        return np.array(y_pred)
    
    def predict_proba(self, X):
        y_proba = []
        
        for x in X:
            # Compute distances
            distances = [self._compute_distance(x, x_train) 
                        for x_train in self.X_train]
            
            # Get k nearest neighbors
            k_indices = np.argsort(distances)[:self.k]
            k_nearest_labels = [self.y_train[i] for i in k_indices]
            
            # Calculate probabilities
            unique_labels = np.unique(self.y_train)
            probabilities = []
            
            for label in unique_labels:
                prob = k_nearest_labels.count(label) / self.k
                probabilities.append(prob)
            
            y_proba.append(probabilities)
        
        return np.array(y_proba)
```

### 3. Distance Metrics Comparison
```python
def compare_distance_metrics(X, y):
    metrics = ['euclidean', 'manhattan', 'minkowski']
    plt.figure(figsize=(15, 5))
    
    for i, metric in enumerate(metrics, 1):
        knn = KNN(k=5, distance_metric=metric)
        knn.fit(X, y)
        
        plt.subplot(1, 3, i)
        plot_decision_boundary(X, y, knn, f'{metric.capitalize()} Distance')
    
    plt.tight_layout()
    plt.show()
```

## Model Evaluation

### 1. Finding Optimal K
```python
def find_optimal_k(X_train, X_test, y_train, y_test, max_k=20):
    k_values = range(1, max_k + 1)
    accuracies = []
    
    for k in k_values:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        accuracies.append(accuracy)
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, accuracies, 'bo-')
    plt.xlabel('K Value')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs K Value')
    plt.grid(True)
    plt.show()
    
    # Return optimal k
    optimal_k = k_values[np.argmax(accuracies)]
    return optimal_k
```

### 2. Cross-validation
```python
from sklearn.model_selection import cross_val_score

def evaluate_knn(X, y, k_values=range(1, 21)):
    cv_scores = []
    
    for k in k_values:
        knn = KNeighborsClassifier(n_neighbors=k)
        scores = cross_val_score(knn, X, y, cv=5)
        cv_scores.append(scores.mean())
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, cv_scores, 'ro-')
    plt.xlabel('K Value')
    plt.ylabel('Cross-validation Score')
    plt.title('Cross-validation Score vs K Value')
    plt.grid(True)
    plt.show()
    
    # Return optimal k
    optimal_k = k_values[np.argmax(cv_scores)]
    return optimal_k
```

## Common Interview Questions

1. **Q: How do you choose the value of K?**
   - A: The choice of K depends on the data:
     - Small K: More sensitive to noise, may overfit
     - Large K: More stable but may underfit
     - Common approach: Use cross-validation to find optimal K
     - Rule of thumb: K = √n, where n is the number of samples

2. **Q: What are the advantages and disadvantages of KNN?**
   - A: Advantages:
     - Simple to understand and implement
     - No training phase
     - Works for both classification and regression
     - Adapts to new data automatically
     Disadvantages:
     - Computationally expensive for large datasets
     - Sensitive to feature scaling
     - Requires careful choice of K
     - Memory intensive

3. **Q: How do you handle different distance metrics in KNN?**
   - A: Common distance metrics include:
     - Euclidean: \(d(x,y) = \sqrt{\sum_{i=1}^n (x_i - y_i)^2}\)
     - Manhattan: \(d(x,y) = \sum_{i=1}^n |x_i - y_i|\)
     - Minkowski: \(d(x,y) = (\sum_{i=1}^n |x_i - y_i|^p)^{1/p}\)
     - Cosine: \(d(x,y) = 1 - \frac{x \cdot y}{||x|| ||y||}\)

## Hands-on Task: Image Classification

### Project: Digit Recognition with KNN
```python
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
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

# Find optimal k
optimal_k = find_optimal_k(X_train_scaled, X_test_scaled, 
                          y_train, y_test, max_k=20)

# Train model with optimal k
knn = KNeighborsClassifier(n_neighbors=optimal_k)
knn.fit(X_train_scaled, y_train)

# Make predictions
y_pred = knn.predict(X_test_scaled)

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
1. Learn about different distance metrics
2. Study advanced KNN variants (weighted KNN)
3. Explore dimensionality reduction techniques
4. Practice with real-world datasets
5. Learn about model optimization

## Resources
- [Scikit-learn KNN](https://scikit-learn.org/stable/modules/neighbors.html)
- [Introduction to KNN](https://www.coursera.org/learn/k-nearest-neighbors)
- [KNN in Python](https://www.kaggle.com/learn/k-nearest-neighbors)
- [Pattern Recognition and Machine Learning](https://www.amazon.com/Pattern-Recognition-Learning-Information-Statistics/dp/0387310738) 