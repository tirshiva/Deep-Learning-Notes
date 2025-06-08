# Anomaly Detection

## Background and Introduction
Anomaly Detection is a technique used to identify unusual patterns that do not conform to expected behavior. It's crucial for detecting fraud, network intrusions, system failures, and other rare events. This technique is widely used in cybersecurity, finance, healthcare, and industrial monitoring.

## What is Anomaly Detection?
Anomaly Detection involves:
1. Identifying rare events
2. Detecting outliers
3. Finding unusual patterns
4. Monitoring system behavior
5. Preventing potential issues

## Why Anomaly Detection?
1. **Fraud Detection**: Identify suspicious transactions
2. **Network Security**: Detect intrusions
3. **Quality Control**: Find defective products
4. **Health Monitoring**: Detect medical anomalies
5. **System Maintenance**: Predict failures

## How to Detect Anomalies?

### 1. Isolation Forest
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_blobs

def create_sample_data(n_samples=1000, n_features=2, contamination=0.1):
    # Generate sample data with outliers
    X, _ = make_blobs(n_samples=n_samples, n_features=n_features, random_state=42)
    
    # Add some outliers
    n_outliers = int(n_samples * contamination)
    outlier_indices = np.random.choice(n_samples, n_outliers, replace=False)
    X[outlier_indices] = np.random.uniform(low=-10, high=10, size=(n_outliers, n_features))
    
    return X, outlier_indices

def detect_anomalies_isolation_forest(X, contamination=0.1):
    # Standardize data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Fit Isolation Forest
    iso_forest = IsolationForest(contamination=contamination, random_state=42)
    predictions = iso_forest.fit_predict(X_scaled)
    
    # Convert predictions to binary (1 for normal, -1 for anomaly)
    is_anomaly = predictions == -1
    
    return {
        'predictions': predictions,
        'is_anomaly': is_anomaly,
        'scaler': scaler,
        'model': iso_forest
    }

def plot_anomalies(X, is_anomaly):
    plt.figure(figsize=(10, 8))
    
    # Plot normal points
    plt.scatter(X[~is_anomaly, 0], X[~is_anomaly, 1],
                c='blue', label='Normal', alpha=0.5)
    
    # Plot anomalies
    plt.scatter(X[is_anomaly, 0], X[is_anomaly, 1],
                c='red', label='Anomaly', alpha=0.5)
    
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Anomaly Detection Results')
    plt.legend()
    plt.show()

def demonstrate_isolation_forest():
    # Create sample data
    X, true_outliers = create_sample_data()
    
    # Detect anomalies
    results = detect_anomalies_isolation_forest(X)
    
    # Plot results
    plot_anomalies(X, results['is_anomaly'])
    
    # Calculate accuracy
    accuracy = np.mean(results['is_anomaly'] == (np.arange(len(X)) in true_outliers))
    print(f"Detection Accuracy: {accuracy:.2f}")
    
    return results
```

### 2. One-Class SVM
```python
from sklearn.svm import OneClassSVM

def detect_anomalies_one_class_svm(X, nu=0.1):
    # Standardize data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Fit One-Class SVM
    svm = OneClassSVM(nu=nu, kernel='rbf', random_state=42)
    predictions = svm.fit_predict(X_scaled)
    
    # Convert predictions to binary (1 for normal, -1 for anomaly)
    is_anomaly = predictions == -1
    
    return {
        'predictions': predictions,
        'is_anomaly': is_anomaly,
        'scaler': scaler,
        'model': svm
    }

def compare_methods(X):
    # Compare Isolation Forest and One-Class SVM
    iso_results = detect_anomalies_isolation_forest(X)
    svm_results = detect_anomalies_one_class_svm(X)
    
    # Plot results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot Isolation Forest results
    ax1.scatter(X[~iso_results['is_anomaly'], 0],
                X[~iso_results['is_anomaly'], 1],
                c='blue', label='Normal', alpha=0.5)
    ax1.scatter(X[iso_results['is_anomaly'], 0],
                X[iso_results['is_anomaly'], 1],
                c='red', label='Anomaly', alpha=0.5)
    ax1.set_title('Isolation Forest')
    ax1.legend()
    
    # Plot One-Class SVM results
    ax2.scatter(X[~svm_results['is_anomaly'], 0],
                X[~svm_results['is_anomaly'], 1],
                c='blue', label='Normal', alpha=0.5)
    ax2.scatter(X[svm_results['is_anomaly'], 0],
                X[svm_results['is_anomaly'], 1],
                c='red', label='Anomaly', alpha=0.5)
    ax2.set_title('One-Class SVM')
    ax2.legend()
    
    plt.tight_layout()
    plt.show()
    
    return {
        'isolation_forest': iso_results,
        'one_class_svm': svm_results
    }
```

### 3. Local Outlier Factor
```python
from sklearn.neighbors import LocalOutlierFactor

def detect_anomalies_lof(X, n_neighbors=20, contamination=0.1):
    # Standardize data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Fit Local Outlier Factor
    lof = LocalOutlierFactor(n_neighbors=n_neighbors,
                            contamination=contamination)
    predictions = lof.fit_predict(X_scaled)
    
    # Convert predictions to binary (1 for normal, -1 for anomaly)
    is_anomaly = predictions == -1
    
    return {
        'predictions': predictions,
        'is_anomaly': is_anomaly,
        'scaler': scaler,
        'model': lof
    }

def plot_anomaly_scores(X, model):
    # Calculate anomaly scores
    scores = -model.negative_outlier_factor_
    
    # Plot anomaly scores
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X[:, 0], X[:, 1], c=scores, cmap='viridis')
    plt.colorbar(scatter, label='Anomaly Score')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Anomaly Scores')
    plt.show()
```

## Model Evaluation

### 1. Performance Metrics
```python
def evaluate_anomaly_detection(true_outliers, predictions):
    # Calculate metrics
    true_positives = np.sum((predictions == -1) & (true_outliers))
    false_positives = np.sum((predictions == -1) & (~true_outliers))
    true_negatives = np.sum((predictions == 1) & (~true_outliers))
    false_negatives = np.sum((predictions == 1) & (true_outliers))
    
    # Calculate metrics
    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)
    f1_score = 2 * (precision * recall) / (precision + recall)
    
    metrics = {
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score
    }
    
    print("\nAnomaly Detection Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    return metrics
```

### 2. Visualization Tools
```python
def visualize_anomaly_detection(X, true_outliers, predictions):
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot true outliers
    ax1.scatter(X[~true_outliers, 0], X[~true_outliers, 1],
                c='blue', label='Normal', alpha=0.5)
    ax1.scatter(X[true_outliers, 0], X[true_outliers, 1],
                c='red', label='True Anomaly', alpha=0.5)
    ax1.set_title('True Anomalies')
    ax1.legend()
    
    # Plot predicted outliers
    ax2.scatter(X[predictions == 1, 0], X[predictions == 1, 1],
                c='blue', label='Normal', alpha=0.5)
    ax2.scatter(X[predictions == -1, 0], X[predictions == -1, 1],
                c='red', label='Predicted Anomaly', alpha=0.5)
    ax2.set_title('Predicted Anomalies')
    ax2.legend()
    
    plt.tight_layout()
    plt.show()
```

## Common Interview Questions

1. **Q: What is the difference between Isolation Forest and One-Class SVM?**
   - A: Isolation Forest is based on random forests and works by isolating anomalies in fewer steps, while One-Class SVM creates a boundary around normal data points. Isolation Forest is generally faster and works well with high-dimensional data, while One-Class SVM can capture complex boundaries but is more computationally expensive.

2. **Q: How do you choose the contamination parameter?**
   - A: The contamination parameter should be chosen based on:
     - Domain knowledge about expected anomaly rate
     - Historical data analysis
     - Business requirements
     - Available labeled data
     - Cross-validation results

3. **Q: What are the advantages and disadvantages of Local Outlier Factor?**
   - A: Advantages:
     - Works well with local density variations
     - Can detect both global and local anomalies
     - No assumptions about data distribution
     Disadvantages:
     - Computationally expensive for large datasets
     - Sensitive to parameter settings
     - May not work well with high-dimensional data

## Hands-on Task: Credit Card Fraud Detection

### Project: Fraud Detection System
```python
def detect_credit_card_fraud():
    # Create sample credit card transaction data
    n_samples = 1000
    n_features = 10
    
    # Generate normal transactions
    X_normal = np.random.normal(0, 1, (n_samples, n_features))
    
    # Generate fraudulent transactions
    n_fraud = int(n_samples * 0.05)  # 5% fraud rate
    X_fraud = np.random.normal(2, 2, (n_fraud, n_features))
    
    # Combine data
    X = np.vstack([X_normal, X_fraud])
    y = np.array([0] * n_samples + [1] * n_fraud)
    
    # Shuffle data
    indices = np.random.permutation(len(X))
    X = X[indices]
    y = y[indices]
    
    # Detect anomalies using different methods
    iso_results = detect_anomalies_isolation_forest(X)
    svm_results = detect_anomalies_one_class_svm(X)
    lof_results = detect_anomalies_lof(X)
    
    # Evaluate results
    print("\nIsolation Forest Results:")
    evaluate_anomaly_detection(y, iso_results['predictions'])
    
    print("\nOne-Class SVM Results:")
    evaluate_anomaly_detection(y, svm_results['predictions'])
    
    print("\nLocal Outlier Factor Results:")
    evaluate_anomaly_detection(y, lof_results['predictions'])
    
    # Visualize results
    visualize_anomaly_detection(X, y, iso_results['predictions'])
    
    return {
        'data': X,
        'labels': y,
        'iso_results': iso_results,
        'svm_results': svm_results,
        'lof_results': lof_results
    }
```

## Next Steps
1. Learn about other anomaly detection algorithms
2. Study time series anomaly detection
3. Explore deep learning approaches
4. Practice with real-world datasets
5. Learn about ensemble methods for anomaly detection

## Resources
- [Scikit-learn Anomaly Detection](https://scikit-learn.org/stable/modules/outlier_detection.html)
- [Isolation Forest Paper](https://cs.nju.edu.cn/zhouzh/zhouzh.files/publication/icdm08b.pdf)
- [Anomaly Detection Tutorial](https://towardsdatascience.com/anomaly-detection-with-isolation-forest-visualization-23cd75c281e2)
- [Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud) 