# Model Evaluation and Selection

## Background and Introduction
Model evaluation and selection are critical steps in the machine learning pipeline. They help us assess model performance, compare different models, and choose the best model for our specific problem. This process involves various metrics, techniques, and best practices to ensure reliable and robust model selection.

## What is Model Evaluation and Selection?
Model evaluation is the process of assessing a model's performance using various metrics and techniques. Model selection involves comparing different models and choosing the best one based on evaluation results, considering factors like performance, complexity, and practical constraints.

## Why Model Evaluation and Selection?
1. **Performance Assessment**: Understand how well models perform
2. **Model Comparison**: Compare different models fairly
3. **Overfitting Detection**: Identify and prevent overfitting
4. **Hyperparameter Tuning**: Find optimal model parameters
5. **Practical Considerations**: Balance performance with complexity

## How to Evaluate and Select Models?

### 1. Cross-Validation
```python
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

def perform_cross_validation(model, X, y, cv=5, scoring='accuracy'):
    # Perform cross-validation
    cv_scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
    
    # Print results
    print(f"Cross-validation scores: {cv_scores}")
    print(f"Mean CV score: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.boxplot(cv_scores)
    plt.title(f'Cross-validation Scores ({scoring})')
    plt.ylabel('Score')
    plt.show()
    
    return cv_scores

# Example usage with different models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

models = {
    'Logistic Regression': LogisticRegression(),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier()
}

# Compare models using cross-validation
def compare_models(models, X, y, cv=5):
    results = {}
    
    for name, model in models.items():
        print(f"\nEvaluating {name}:")
        scores = perform_cross_validation(model, X, y, cv=cv)
        results[name] = scores
    
    # Plot comparison
    plt.figure(figsize=(12, 6))
    plt.boxplot([results[name] for name in models.keys()],
                labels=list(models.keys()))
    plt.title('Model Comparison')
    plt.ylabel('Cross-validation Score')
    plt.xticks(rotation=45)
    plt.show()
    
    return results
```

### 2. Performance Metrics
```python
def evaluate_classification_metrics(y_true, y_pred, y_prob=None):
    metrics = {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred, average='weighted'),
        'Recall': recall_score(y_true, y_pred, average='weighted'),
        'F1 Score': f1_score(y_true, y_pred, average='weighted')
    }
    
    # Print metrics
    for metric, value in metrics.items():
        print(f"{metric}: {value:.3f}")
    
    # Plot confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()
    
    # Plot ROC curve if probabilities are provided
    if y_prob is not None:
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.show()
    
    return metrics

def evaluate_regression_metrics(y_true, y_pred):
    metrics = {
        'MSE': mean_squared_error(y_true, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'MAE': mean_absolute_error(y_true, y_pred),
        'R2': r2_score(y_true, y_pred)
    }
    
    # Print metrics
    for metric, value in metrics.items():
        print(f"{metric}: {value:.3f}")
    
    # Plot actual vs predicted
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title('Actual vs Predicted')
    plt.show()
    
    return metrics
```

### 3. Learning Curves
```python
def plot_learning_curves(model, X, y, cv=5):
    train_sizes, train_scores, val_scores = learning_curve(
        model, X, y, cv=cv, n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 10),
        scoring='accuracy'
    )
    
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_mean, label='Training score')
    plt.plot(train_sizes, val_mean, label='Cross-validation score')
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1)
    plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1)
    plt.xlabel('Training Examples')
    plt.ylabel('Score')
    plt.title('Learning Curves')
    plt.legend(loc='best')
    plt.grid(True)
    plt.show()
```

## Model Selection Techniques

### 1. Grid Search
```python
from sklearn.model_selection import GridSearchCV

def perform_grid_search(model, param_grid, X, y, cv=5):
    grid_search = GridSearchCV(
        model, param_grid, cv=cv, scoring='accuracy',
        n_jobs=-1, verbose=1
    )
    grid_search.fit(X, y)
    
    print("Best parameters:", grid_search.best_params_)
    print("Best cross-validation score:", grid_search.best_score_)
    
    # Plot results
    results = pd.DataFrame(grid_search.cv_results_)
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=results, x='param_C', y='mean_test_score')
    plt.title('Grid Search Results')
    plt.xlabel('C parameter')
    plt.ylabel('Mean CV Score')
    plt.show()
    
    return grid_search
```

### 2. Model Complexity Analysis
```python
def analyze_model_complexity(model, X, y, param_name, param_range):
    train_scores = []
    val_scores = []
    
    for param in param_range:
        model.set_params(**{param_name: param})
        model.fit(X, y)
        train_scores.append(model.score(X, y))
        val_scores.append(cross_val_score(model, X, y, cv=5).mean())
    
    plt.figure(figsize=(10, 6))
    plt.plot(param_range, train_scores, label='Training score')
    plt.plot(param_range, val_scores, label='Cross-validation score')
    plt.xlabel(param_name)
    plt.ylabel('Score')
    plt.title('Model Complexity Analysis')
    plt.legend()
    plt.grid(True)
    plt.show()
```

## Common Interview Questions

1. **Q: What is the difference between validation and test sets?**
   - A: The validation set is used to tune hyperparameters and select the best model, while the test set is used for final evaluation. The test set should only be used once, at the very end of the process.

2. **Q: How do you handle imbalanced datasets in model evaluation?**
   - A: Several approaches can be used:
     - Use appropriate metrics (precision, recall, F1-score)
     - Stratified sampling in cross-validation
     - Resampling techniques (oversampling, undersampling)
     - Class weights in model training
     - ROC and PR curves

3. **Q: What are the advantages and disadvantages of different cross-validation techniques?**
   - A: Common techniques include:
     - K-Fold: Good for balanced datasets
     - Stratified K-Fold: Better for imbalanced datasets
     - Leave-One-Out: Computationally expensive
     - Time Series CV: For temporal data
     - Nested CV: For both model selection and evaluation

## Hands-on Task: Model Selection Project

### Project: Credit Card Fraud Detection
```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# Load credit card fraud dataset
# Note: Replace with actual data loading code
# data = pd.read_csv('creditcard.csv')

# Create sample data
np.random.seed(42)
n_samples = 1000
n_features = 30
X = np.random.randn(n_samples, n_features)
y = np.random.randint(0, 2, n_samples)  # Binary classification

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define models to compare
models = {
    'Random Forest': RandomForestClassifier(),
    'Logistic Regression': LogisticRegression(),
    'SVM': SVC(probability=True)
}

# Compare models
results = compare_models(models, X_train_scaled, y_train)

# Select best model
best_model_name = max(results, key=lambda k: np.mean(results[k]))
best_model = models[best_model_name]

# Train best model
best_model.fit(X_train_scaled, y_train)

# Evaluate on test set
y_pred = best_model.predict(X_test_scaled)
y_prob = best_model.predict_proba(X_test_scaled)[:, 1]

# Print evaluation metrics
print("\nBest Model Evaluation:")
evaluate_classification_metrics(y_test, y_pred, y_prob)

# Plot learning curves
plot_learning_curves(best_model, X_train_scaled, y_train)

# Perform grid search for hyperparameter tuning
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10]
}

grid_search = perform_grid_search(best_model, param_grid, X_train_scaled, y_train)

# Train final model with best parameters
final_model = grid_search.best_estimator_
final_model.fit(X_train_scaled, y_train)

# Evaluate final model
y_pred_final = final_model.predict(X_test_scaled)
y_prob_final = final_model.predict_proba(X_test_scaled)[:, 1]

print("\nFinal Model Evaluation:")
evaluate_classification_metrics(y_test, y_pred_final, y_prob_final)
```

## Next Steps
1. Learn about advanced evaluation techniques
2. Study model interpretability methods
3. Explore ensemble methods
4. Practice with real-world datasets
5. Learn about model deployment

## Resources
- [Scikit-learn Model Selection](https://scikit-learn.org/stable/model_selection.html)
- [Introduction to Model Evaluation](https://www.coursera.org/learn/model-evaluation)
- [Model Selection in Python](https://www.kaggle.com/learn/model-selection)
- [Elements of Statistical Learning](https://www.amazon.com/Elements-Statistical-Learning-Prediction-Statistics/dp/0387848576) 