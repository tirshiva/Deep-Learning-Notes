# Decision Trees and Random Forests

## Background and Introduction
Decision Trees and Random Forests are powerful supervised learning algorithms that can be used for both classification and regression tasks. Decision Trees were first introduced in the 1960s, while Random Forests were developed in the 1990s as an ensemble method to improve upon single decision trees.

## What are Decision Trees?
A Decision Tree is a flowchart-like structure where each internal node represents a feature test, each branch represents the outcome of the test, and each leaf node represents a class label or prediction. The tree is built by recursively splitting the data based on feature values.

## Why Decision Trees and Random Forests?
1. **Interpretability**: Easy to understand and visualize
2. **Handles Non-linear Data**: Can capture complex patterns
3. **Feature Importance**: Provides insights into feature relationships
4. **Robustness**: Random Forests reduce overfitting
5. **Versatility**: Works with both numerical and categorical data

## How Do Decision Trees Work?

### 1. Decision Tree Basics
```python
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

# Generate sample data
np.random.seed(42)
X = np.random.randn(100, 2)
y = (X[:, 0] + X[:, 1] > 0).astype(int)

# Create and train decision tree
dt = DecisionTreeClassifier(max_depth=3)
dt.fit(X, y)

# Visualize the tree
plt.figure(figsize=(20,10))
plot_tree(dt, feature_names=['Feature 1', 'Feature 2'],
          class_names=['Class 0', 'Class 1'],
          filled=True, rounded=True)
plt.show()
```

### 2. Decision Tree Implementation from Scratch
```python
class DecisionTree:
    def __init__(self, max_depth=None, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None
        
    class Node:
        def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
            self.feature = feature
            self.threshold = threshold
            self.left = left
            self.right = right
            self.value = value
    
    def _gini_impurity(self, y):
        classes = np.unique(y)
        impurity = 1
        for c in classes:
            p = len(y[y == c]) / len(y)
            impurity -= p**2
        return impurity
    
    def _best_split(self, X, y):
        best_gain = -1
        best_feature = None
        best_threshold = None
        
        parent_impurity = self._gini_impurity(y)
        
        for feature in range(X.shape[1]):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                left_mask = X[:, feature] <= threshold
                right_mask = ~left_mask
                
                if len(y[left_mask]) == 0 or len(y[right_mask]) == 0:
                    continue
                
                left_impurity = self._gini_impurity(y[left_mask])
                right_impurity = self._gini_impurity(y[right_mask])
                
                n = len(y)
                left_n = len(y[left_mask])
                right_n = len(y[right_mask])
                
                gain = parent_impurity - (left_n/n * left_impurity + right_n/n * right_impurity)
                
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold
        
        return best_feature, best_threshold
    
    def _build_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))
        
        # Stopping criteria
        if (self.max_depth is not None and depth >= self.max_depth) or \
           n_samples < self.min_samples_split or \
           n_classes == 1:
            leaf_value = np.argmax(np.bincount(y))
            return self.Node(value=leaf_value)
        
        # Find best split
        best_feature, best_threshold = self._best_split(X, y)
        
        if best_feature is None:
            leaf_value = np.argmax(np.bincount(y))
            return self.Node(value=leaf_value)
        
        # Create child nodes
        left_mask = X[:, best_feature] <= best_threshold
        right_mask = ~left_mask
        
        left = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right = self._build_tree(X[right_mask], y[right_mask], depth + 1)
        
        return self.Node(best_feature, best_threshold, left, right)
    
    def fit(self, X, y):
        self.root = self._build_tree(X, y)
    
    def _traverse_tree(self, x, node):
        if node.value is not None:
            return node.value
        
        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)
    
    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])
```

### 3. Random Forest Implementation
```python
class RandomForest:
    def __init__(self, n_trees=100, max_depth=None, min_samples_split=2):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.trees = []
        
    def _bootstrap_sample(self, X, y):
        n_samples = X.shape[0]
        idxs = np.random.choice(n_samples, n_samples, replace=True)
        return X[idxs], y[idxs]
    
    def fit(self, X, y):
        self.trees = []
        for _ in range(self.n_trees):
            tree = DecisionTree(max_depth=self.max_depth,
                              min_samples_split=self.min_samples_split)
            X_sample, y_sample = self._bootstrap_sample(X, y)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)
    
    def predict(self, X):
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        return np.array([np.bincount(tree_preds[:, i]).argmax() 
                        for i in range(len(X))])
```

## Model Evaluation

### 1. Decision Tree Visualization
```python
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

# Compare Decision Tree and Random Forest
dt = DecisionTreeClassifier(max_depth=3)
rf = RandomForestClassifier(n_estimators=100, max_depth=3)

dt.fit(X, y)
rf.fit(X, y)

plot_decision_boundary(X, y, dt, 'Decision Tree Decision Boundary')
plot_decision_boundary(X, y, rf, 'Random Forest Decision Boundary')
```

### 2. Feature Importance
```python
def plot_feature_importance(model, feature_names):
    if isinstance(model, RandomForestClassifier):
        importances = model.feature_importances_
    else:
        importances = model.feature_importances_
    
    indices = np.argsort(importances)[::-1]
    
    plt.figure(figsize=(10, 6))
    plt.title('Feature Importance')
    plt.bar(range(len(importances)), importances[indices])
    plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45)
    plt.tight_layout()
    plt.show()
```

## Common Interview Questions

1. **Q: What is the difference between Decision Trees and Random Forests?**
   - A: Decision Trees are single models that make decisions based on feature splits, while Random Forests are ensembles of multiple decision trees that vote on the final prediction. Random Forests reduce overfitting and improve generalization.

2. **Q: How do you handle overfitting in Decision Trees?**
   - A: Several techniques can be used:
     - Pruning (pre-pruning and post-pruning)
     - Setting maximum depth
     - Minimum samples per leaf
     - Using Random Forests
     - Cross-validation

3. **Q: What are the advantages and disadvantages of Random Forests?**
   - A: Advantages:
     - Reduces overfitting
     - Handles non-linear relationships
     - Provides feature importance
     - Works well with high-dimensional data
     Disadvantages:
     - Can be computationally expensive
     - Less interpretable than single trees
     - May require more memory

## Hands-on Task: Credit Risk Assessment

### Project: Loan Default Prediction
```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# Create sample loan data
np.random.seed(42)
n_loans = 1000
data = {
    'income': np.random.normal(50000, 20000, n_loans),
    'credit_score': np.random.normal(700, 50, n_loans),
    'debt_to_income': np.random.normal(0.3, 0.1, n_loans),
    'employment_length': np.random.normal(5, 2, n_loans),
    'default': np.random.randint(0, 2, n_loans)
}
df = pd.DataFrame(data)

# Prepare features and target
X = df.drop('default', axis=1)
y = df['default']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Random Forest
rf = RandomForestClassifier(n_estimators=100, max_depth=5)
rf.fit(X_train_scaled, y_train)

# Make predictions
y_pred = rf.predict(X_test_scaled)

# Evaluate model
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Plot confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Feature importance
plot_feature_importance(rf, X.columns)

# Visualize decision boundaries for pairs of features
def plot_feature_pairs(X, y, model, feature_pairs):
    for pair in feature_pairs:
        plt.figure(figsize=(10, 6))
        X_pair = X[:, pair]
        h = 0.02
        x_min, x_max = X_pair[:, 0].min() - 1, X_pair[:, 0].max() + 1
        y_min, y_max = X_pair[:, 1].min() - 1, X_pair[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                            np.arange(y_min, y_max, h))
        
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        
        plt.contourf(xx, yy, Z, alpha=0.4)
        plt.scatter(X_pair[:, 0], X_pair[:, 1], c=y, alpha=0.8)
        plt.title(f'Decision Boundary: {X.columns[pair[0]]} vs {X.columns[pair[1]]}')
        plt.xlabel(X.columns[pair[0]])
        plt.ylabel(X.columns[pair[1]])
        plt.show()

# Plot decision boundaries for important feature pairs
feature_pairs = [(0, 1), (0, 2), (1, 2)]  # income vs credit_score, income vs debt_to_income, etc.
plot_feature_pairs(X_train_scaled, y_train, rf, feature_pairs)
```

## Next Steps
1. Learn about other ensemble methods (Bagging, Boosting)
2. Study advanced tree-based algorithms (XGBoost, LightGBM)
3. Explore hyperparameter tuning techniques
4. Practice with real-world datasets
5. Learn about model interpretation tools

## Resources
- [Scikit-learn Decision Trees](https://scikit-learn.org/stable/modules/tree.html)
- [Random Forests in Python](https://scikit-learn.org/stable/modules/ensemble.html#forest)
- [Introduction to Decision Trees](https://www.coursera.org/learn/decision-trees)
- [Tree-based Methods for Statistical Learning](https://www.amazon.com/Elements-Statistical-Learning-Prediction-Statistics/dp/0387848576) 