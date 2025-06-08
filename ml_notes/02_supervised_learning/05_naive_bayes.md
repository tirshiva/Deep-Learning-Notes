# Naive Bayes

## Background and Introduction
Naive Bayes is a family of probabilistic classifiers based on Bayes' theorem with strong (naive) independence assumptions between features. Despite its simplicity, it often performs well in practice and is particularly effective for text classification and spam filtering.

## What is Naive Bayes?
Naive Bayes is a classification algorithm that applies Bayes' theorem with the assumption that all features are independent of each other. It calculates the probability of a class given a set of features using the formula:

\[ P(y|X) = \frac{P(X|y)P(y)}{P(X)} \]

where:
- \(P(y|X)\) is the posterior probability
- \(P(X|y)\) is the likelihood
- \(P(y)\) is the prior probability
- \(P(X)\) is the evidence

## Why Naive Bayes?
1. **Fast Training and Prediction**: Simple calculations make it very efficient
2. **Works Well with High Dimensions**: Effective even with many features
3. **Requires Little Data**: Can work with small training sets
4. **Handles Missing Data**: Can work with incomplete features
5. **Probabilistic Output**: Provides probability estimates

## How Does Naive Bayes Work?

### 1. Types of Naive Bayes
```python
import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Generate sample data
np.random.seed(42)
n_samples = 1000
X = np.random.randn(n_samples, 2)
y = (X[:, 0] + X[:, 1] > 0).astype(int)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Compare different Naive Bayes implementations
models = {
    'GaussianNB': GaussianNB(),
    'MultinomialNB': MultinomialNB(),
    'BernoulliNB': BernoulliNB()
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    results[name] = accuracy

# Plot results
plt.figure(figsize=(10, 6))
plt.bar(results.keys(), results.values())
plt.title('Naive Bayes Models Comparison')
plt.ylabel('Accuracy')
plt.show()
```

### 2. Implementation from Scratch
```python
class NaiveBayes:
    def __init__(self, var_smoothing=1e-9):
        self.var_smoothing = var_smoothing
        self.classes = None
        self.mean = None
        self.var = None
        self.priors = None
        
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.classes = np.unique(y)
        n_classes = len(self.classes)
        
        # Initialize parameters
        self.mean = np.zeros((n_classes, n_features))
        self.var = np.zeros((n_classes, n_features))
        self.priors = np.zeros(n_classes)
        
        # Calculate parameters for each class
        for idx, c in enumerate(self.classes):
            X_c = X[y == c]
            self.mean[idx, :] = X_c.mean(axis=0)
            self.var[idx, :] = X_c.var(axis=0) + self.var_smoothing
            self.priors[idx] = X_c.shape[0] / n_samples
    
    def _calculate_likelihood(self, X, mean, var):
        return (1 / np.sqrt(2 * np.pi * var)) * \
               np.exp(-((X - mean) ** 2) / (2 * var))
    
    def predict(self, X):
        y_pred = []
        
        for x in X:
            posteriors = []
            
            # Calculate posterior probability for each class
            for idx, c in enumerate(self.classes):
                prior = np.log(self.priors[idx])
                likelihood = np.sum(np.log(self._calculate_likelihood(x, 
                                                                    self.mean[idx], 
                                                                    self.var[idx])))
                posterior = prior + likelihood
                posteriors.append(posterior)
            
            # Get class with highest posterior probability
            y_pred.append(self.classes[np.argmax(posteriors)])
        
        return np.array(y_pred)
    
    def predict_proba(self, X):
        y_proba = []
        
        for x in X:
            posteriors = []
            
            # Calculate posterior probability for each class
            for idx, c in enumerate(self.classes):
                prior = np.log(self.priors[idx])
                likelihood = np.sum(np.log(self._calculate_likelihood(x, 
                                                                    self.mean[idx], 
                                                                    self.var[idx])))
                posterior = prior + likelihood
                posteriors.append(posterior)
            
            # Convert to probabilities
            posteriors = np.exp(posteriors)
            posteriors = posteriors / np.sum(posteriors)
            y_proba.append(posteriors)
        
        return np.array(y_proba)
```

### 3. Text Classification Example
```python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

# Sample text data
texts = [
    "I love this product, it's amazing!",
    "This is the worst purchase ever",
    "Great service and fast delivery",
    "Terrible customer support",
    "The quality is excellent",
    "I'm very disappointed with this"
]
labels = [1, 0, 1, 0, 1, 0]  # 1 for positive, 0 for negative

# Create pipeline
pipeline = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('classifier', MultinomialNB())
])

# Train model
pipeline.fit(texts, labels)

# Make predictions
new_texts = [
    "This product is fantastic",
    "I hate this service"
]
predictions = pipeline.predict(new_texts)
probabilities = pipeline.predict_proba(new_texts)

# Print results
for text, pred, prob in zip(new_texts, predictions, probabilities):
    print(f"\nText: {text}")
    print(f"Prediction: {'Positive' if pred == 1 else 'Negative'}")
    print(f"Probability: {prob[pred]:.2f}")
```

## Model Evaluation

### 1. Performance Metrics
```python
def evaluate_naive_bayes(X_train, X_test, y_train, y_test):
    # Train model
    model = GaussianNB()
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)
    
    # Calculate metrics
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
    
    # Plot ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_prob[:, 1])
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.show()
```

### 2. Feature Importance
```python
def plot_feature_importance(X, y, feature_names):
    model = GaussianNB()
    model.fit(X, y)
    
    # Calculate feature importance
    importance = np.abs(model.theta_[1] - model.theta_[0])
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.barh(feature_names, importance)
    plt.title('Feature Importance')
    plt.xlabel('Importance')
    plt.show()
```

## Common Interview Questions

1. **Q: Why is it called "Naive" Bayes?**
   - A: It's called "naive" because it makes the assumption that all features are independent of each other, which is often not true in real-world data. This assumption simplifies the calculations but can sometimes lead to suboptimal results.

2. **Q: What are the different types of Naive Bayes classifiers?**
   - A: Common types include:
     - Gaussian Naive Bayes: For continuous features
     - Multinomial Naive Bayes: For discrete features (e.g., word counts)
     - Bernoulli Naive Bayes: For binary features
     - Complement Naive Bayes: For imbalanced datasets

3. **Q: What are the advantages and disadvantages of Naive Bayes?**
   - A: Advantages:
     - Fast training and prediction
     - Works well with high-dimensional data
     - Requires little training data
     - Handles missing data well
     Disadvantages:
     - Assumes feature independence
     - Can be outperformed by more complex models
     - Sensitive to feature scaling
     - May not work well with correlated features

## Hands-on Task: Spam Detection

### Project: Email Spam Classifier
```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# Create sample email data
emails = [
    "Get rich quick! Click here to win a million dollars!",
    "Meeting tomorrow at 10 AM in the conference room",
    "URGENT: Your account needs verification",
    "Project update: All tasks completed on time",
    "Congratulations! You've won a free iPhone",
    "Please review the attached documents",
    "Limited time offer: 50% off all products",
    "Team lunch this Friday at 12 PM"
]
labels = [1, 0, 1, 0, 1, 0, 1, 0]  # 1 for spam, 0 for ham

# Create DataFrame
df = pd.DataFrame({
    'email': emails,
    'label': labels
})

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    df['email'], df['label'], test_size=0.2, random_state=42
)

# Create pipeline
pipeline = Pipeline([
    ('vectorizer', TfidfVectorizer()),
    ('classifier', MultinomialNB())
])

# Train model
pipeline.fit(X_train, y_train)

# Make predictions
y_pred = pipeline.predict(X_test)
y_prob = pipeline.predict_proba(X_test)

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

# Test with new emails
new_emails = [
    "Important: Your password needs to be reset",
    "Team meeting notes from yesterday",
    "You've been selected for a special offer",
    "Please submit your timesheet by Friday"
]

predictions = pipeline.predict(new_emails)
probabilities = pipeline.predict_proba(new_emails)

# Print results
for email, pred, prob in zip(new_emails, predictions, probabilities):
    print(f"\nEmail: {email}")
    print(f"Prediction: {'Spam' if pred == 1 else 'Ham'}")
    print(f"Probability: {prob[pred]:.2f}")
```

## Next Steps
1. Learn about different types of Naive Bayes
2. Study text classification techniques
3. Explore feature engineering for Naive Bayes
4. Practice with real-world datasets
5. Learn about model interpretation

## Resources
- [Scikit-learn Naive Bayes](https://scikit-learn.org/stable/modules/naive_bayes.html)
- [Introduction to Naive Bayes](https://www.coursera.org/learn/naive-bayes)
- [Text Classification with Naive Bayes](https://www.kaggle.com/learn/text-classification)
- [Machine Learning with Naive Bayes](https://www.amazon.com/Machine-Learning-Naive-Bayes-Classification/dp/1542967021) 