# Ensemble Methods

## Background and Introduction
Ensemble methods combine multiple machine learning models to create a more robust and accurate prediction system. The idea is that by combining the predictions of several models, we can reduce variance, bias, and improve overall performance. Ensemble methods have become increasingly popular in machine learning competitions and real-world applications.

## What are Ensemble Methods?
Ensemble methods are techniques that combine multiple base models to create a more powerful model. The main types of ensemble methods are:
1. **Bagging**: Parallel training of models on different subsets of data
2. **Boosting**: Sequential training of models, each focusing on previous errors
3. **Stacking**: Combining predictions of different models using a meta-learner
4. **Voting**: Combining predictions through majority voting or averaging

## Why Ensemble Methods?
1. **Improved Accuracy**: Better performance than individual models
2. **Reduced Variance**: More stable predictions
3. **Robustness**: Less sensitive to noise and outliers
4. **Generalization**: Better performance on unseen data
5. **Error Reduction**: Different models can capture different patterns

## How Do Ensemble Methods Work?

### 1. Bagging (Bootstrap Aggregating)
```python
import numpy as np
import pandas as pd
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

class BaggingEnsemble:
    def __init__(self, base_model, n_estimators=10):
        self.base_model = base_model
        self.n_estimators = n_estimators
        self.models = []
        
    def fit(self, X, y):
        self.models = []
        n_samples = X.shape[0]
        
        for _ in range(self.n_estimators):
            # Create bootstrap sample
            indices = np.random.choice(n_samples, n_samples, replace=True)
            X_bootstrap = X[indices]
            y_bootstrap = y[indices]
            
            # Train model
            model = self.base_model()
            model.fit(X_bootstrap, y_bootstrap)
            self.models.append(model)
    
    def predict(self, X):
        predictions = np.array([model.predict(X) for model in self.models])
        return np.apply_along_axis(
            lambda x: np.argmax(np.bincount(x)), axis=0, arr=predictions
        )
    
    def predict_proba(self, X):
        predictions = np.array([model.predict_proba(X) for model in self.models])
        return np.mean(predictions, axis=0)

# Example usage
def demonstrate_bagging(X, y):
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    # Create and train bagging ensemble
    bagging = BaggingEnsemble(DecisionTreeClassifier, n_estimators=10)
    bagging.fit(X_train, y_train)
    
    # Make predictions
    y_pred = bagging.predict(X_test)
    
    # Evaluate
    print("Bagging Ensemble Results:")
    print(classification_report(y_test, y_pred))
    
    return bagging
```

### 2. Boosting
```python
class AdaBoost:
    def __init__(self, base_model, n_estimators=50, learning_rate=1.0):
        self.base_model = base_model
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.models = []
        self.weights = []
        
    def fit(self, X, y):
        n_samples = X.shape[0]
        sample_weights = np.ones(n_samples) / n_samples
        
        for _ in range(self.n_estimators):
            # Train model with current weights
            model = self.base_model()
            model.fit(X, y, sample_weight=sample_weights)
            
            # Make predictions
            y_pred = model.predict(X)
            
            # Calculate error
            error = np.sum(sample_weights[y_pred != y]) / np.sum(sample_weights)
            
            # Calculate model weight
            model_weight = self.learning_rate * 0.5 * np.log((1 - error) / error)
            
            # Update sample weights
            sample_weights *= np.exp(-model_weight * y * y_pred)
            sample_weights /= np.sum(sample_weights)
            
            # Save model and weight
            self.models.append(model)
            self.weights.append(model_weight)
    
    def predict(self, X):
        predictions = np.array([model.predict(X) for model in self.models])
        weighted_predictions = np.average(predictions, 
                                       weights=self.weights, 
                                       axis=0)
        return np.sign(weighted_predictions)

# Example usage
def demonstrate_boosting(X, y):
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    # Create and train AdaBoost
    boosting = AdaBoost(DecisionTreeClassifier, n_estimators=50)
    boosting.fit(X_train, y_train)
    
    # Make predictions
    y_pred = boosting.predict(X_test)
    
    # Evaluate
    print("AdaBoost Results:")
    print(classification_report(y_test, y_pred))
    
    return boosting
```

### 3. Stacking
```python
class StackingEnsemble:
    def __init__(self, base_models, meta_model):
        self.base_models = base_models
        self.meta_model = meta_model
        self.base_predictions = None
        
    def fit(self, X, y):
        # Train base models
        for model in self.base_models:
            model.fit(X, y)
        
        # Generate base predictions
        self.base_predictions = np.column_stack([
            model.predict_proba(X)[:, 1] for model in self.base_models
        ])
        
        # Train meta-model
        self.meta_model.fit(self.base_predictions, y)
    
    def predict(self, X):
        # Generate base predictions
        base_preds = np.column_stack([
            model.predict_proba(X)[:, 1] for model in self.base_models
        ])
        
        # Make final prediction
        return self.meta_model.predict(base_preds)
    
    def predict_proba(self, X):
        base_preds = np.column_stack([
            model.predict_proba(X)[:, 1] for model in self.base_models
        ])
        return self.meta_model.predict_proba(base_preds)

# Example usage
def demonstrate_stacking(X, y):
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    # Define base models
    base_models = [
        RandomForestClassifier(),
        DecisionTreeClassifier(),
        LogisticRegression()
    ]
    
    # Define meta-model
    meta_model = LogisticRegression()
    
    # Create and train stacking ensemble
    stacking = StackingEnsemble(base_models, meta_model)
    stacking.fit(X_train, y_train)
    
    # Make predictions
    y_pred = stacking.predict(X_test)
    
    # Evaluate
    print("Stacking Ensemble Results:")
    print(classification_report(y_test, y_pred))
    
    return stacking
```

## Model Evaluation

### 1. Ensemble Performance Analysis
```python
def analyze_ensemble_performance(models, X, y):
    results = {}
    
    for name, model in models.items():
        # Perform cross-validation
        scores = cross_val_score(model, X, y, cv=5)
        results[name] = scores
    
    # Plot results
    plt.figure(figsize=(12, 6))
    plt.boxplot([results[name] for name in models.keys()],
                labels=list(models.keys()))
    plt.title('Ensemble Methods Comparison')
    plt.ylabel('Cross-validation Score')
    plt.xticks(rotation=45)
    plt.show()
    
    return results
```

### 2. Feature Importance Analysis
```python
def plot_feature_importance(model, feature_names):
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    else:
        importances = np.mean([m.feature_importances_ 
                             for m in model.estimators_], axis=0)
    
    indices = np.argsort(importances)[::-1]
    
    plt.figure(figsize=(10, 6))
    plt.title('Feature Importance')
    plt.barh(range(len(importances)), importances[indices])
    plt.yticks(range(len(importances)), [feature_names[i] for i in indices])
    plt.show()
```

## Common Interview Questions

1. **Q: What is the difference between bagging and boosting?**
   - A: Key differences:
     - Bagging: Parallel training, equal weights, reduces variance
     - Boosting: Sequential training, adaptive weights, reduces bias
     - Bagging: Independent models, Boosting: Dependent models
     - Bagging: Less prone to overfitting, Boosting: More prone to overfitting

2. **Q: When should you use ensemble methods?**
   - A: Ensemble methods are particularly useful when:
     - Individual models have high variance
     - Different models capture different patterns
     - You have sufficient computational resources
     - You need robust predictions
     - You want to reduce overfitting

3. **Q: What are the advantages and disadvantages of different ensemble methods?**
   - A: Advantages:
     - Improved accuracy and robustness
     - Better generalization
     - Reduced variance and bias
     Disadvantages:
     - Increased computational complexity
     - More difficult to interpret
     - May require more data
     - Can be prone to overfitting

## Hands-on Task: Credit Risk Assessment

### Project: Ensemble Credit Scoring
```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# Create sample credit data
np.random.seed(42)
n_samples = 1000
n_features = 20
X = np.random.randn(n_samples, n_features)
y = np.random.randint(0, 2, n_samples)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define models
models = {
    'Random Forest': RandomForestClassifier(),
    'Gradient Boosting': GradientBoostingClassifier(),
    'Logistic Regression': LogisticRegression()
}

# Compare individual models
print("Individual Model Performance:")
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    print(f"\n{name}:")
    print(classification_report(y_test, y_pred))

# Create stacking ensemble
base_models = [
    RandomForestClassifier(),
    GradientBoostingClassifier(),
    LogisticRegression()
]
meta_model = LogisticRegression()

stacking = StackingEnsemble(base_models, meta_model)
stacking.fit(X_train_scaled, y_train)

# Evaluate stacking ensemble
y_pred_stack = stacking.predict(X_test_scaled)
print("\nStacking Ensemble Performance:")
print(classification_report(y_test, y_pred_stack))

# Plot confusion matrix
cm = confusion_matrix(y_test, y_pred_stack)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix - Stacking Ensemble')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Analyze feature importance
feature_names = [f'Feature_{i}' for i in range(n_features)]
plot_feature_importance(stacking.base_models[0], feature_names)
```

## Next Steps
1. Learn about advanced ensemble techniques
2. Study model interpretability methods
3. Explore deep learning ensembles
4. Practice with real-world datasets
5. Learn about model deployment

## Resources
- [Scikit-learn Ensemble Methods](https://scikit-learn.org/stable/modules/ensemble.html)
- [Introduction to Ensemble Methods](https://www.coursera.org/learn/ensemble-methods)
- [Ensemble Learning in Python](https://www.kaggle.com/learn/ensemble-methods)
- [Ensemble Methods in Machine Learning](https://www.amazon.com/Ensemble-Methods-Learning-Chapelle-Statistical/dp/0262033949) 