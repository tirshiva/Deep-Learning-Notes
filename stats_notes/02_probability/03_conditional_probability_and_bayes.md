# Conditional Probability and Bayes' Theorem

## What are Conditional Probability and Bayes' Theorem?

### Conditional Probability
Conditional probability is the probability of an event occurring given that another event has already occurred. It helps us update our beliefs based on new information.

### Bayes' Theorem
Bayes' Theorem is a fundamental concept that describes how to update the probabilities of hypotheses when given evidence. It's the foundation of many machine learning algorithms.

## Why are They Important in Machine Learning?

1. **Model Uncertainty**
   - Updating predictions with new data
   - Handling incomplete information
   - Quantifying model confidence

2. **Classification Problems**
   - Naive Bayes classifier
   - Spam detection
   - Medical diagnosis

3. **Feature Importance**
   - Understanding feature relationships
   - Feature selection
   - Model interpretation

## How Do They Work?

### 1. Conditional Probability

#### Basic Concept
**Formula**: \[ P(A|B) = \frac{P(A \cap B)}{P(B)} \]

```python
def conditional_probability(prob_a_and_b, prob_b):
    return prob_a_and_b / prob_b

# Example: Probability of rain given cloudy sky
prob_rain_and_cloudy = 0.3
prob_cloudy = 0.4
prob_rain_given_cloudy = conditional_probability(prob_rain_and_cloudy, prob_cloudy)  # 0.75
```

#### Chain Rule
**Formula**: \[ P(A \cap B) = P(A|B) \times P(B) \]

```python
def chain_rule(prob_a_given_b, prob_b):
    return prob_a_given_b * prob_b

# Example: Probability of being sick and having a fever
prob_sick_given_fever = 0.7
prob_fever = 0.2
prob_sick_and_fever = chain_rule(prob_sick_given_fever, prob_fever)  # 0.14
```

### 2. Bayes' Theorem

#### Basic Form
**Formula**: \[ P(A|B) = \frac{P(B|A) \times P(A)}{P(B)} \]

```python
def bayes_theorem(prob_b_given_a, prob_a, prob_b):
    return (prob_b_given_a * prob_a) / prob_b

# Example: Medical test
prob_positive_given_disease = 0.95  # Test sensitivity
prob_disease = 0.01  # Disease prevalence
prob_positive = 0.05  # Test positive rate
prob_disease_given_positive = bayes_theorem(
    prob_positive_given_disease,
    prob_disease,
    prob_positive
)  # 0.19
```

#### Law of Total Probability
**Formula**: \[ P(B) = \sum_{i} P(B|A_i) \times P(A_i) \]

```python
def total_probability(prob_b_given_a_list, prob_a_list):
    return sum(p_b_given_a * p_a for p_b_given_a, p_a in zip(prob_b_given_a_list, prob_a_list))

# Example: Probability of test being positive
prob_positive_given_disease = 0.95
prob_positive_given_no_disease = 0.05
prob_disease = 0.01
prob_no_disease = 0.99
prob_positive = total_probability(
    [prob_positive_given_disease, prob_positive_given_no_disease],
    [prob_disease, prob_no_disease]
)  # 0.059
```

## Visualizing Conditional Probability

```python
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def plot_conditional_probability(prob_a, prob_b_given_a, prob_b_given_not_a):
    # Create probability tree
    plt.figure(figsize=(10, 6))
    
    # Plot probabilities
    x = [0, 1, 1]
    y = [0, 1, -1]
    labels = [f'P(A) = {prob_a:.2f}',
              f'P(B|A) = {prob_b_given_a:.2f}',
              f'P(B|¬A) = {prob_b_given_not_a:.2f}']
    
    plt.plot(x, y, 'b-')
    plt.scatter(x, y, c='red', s=100)
    
    for i, label in enumerate(labels):
        plt.text(x[i] + 0.1, y[i], label)
    
    plt.title('Probability Tree')
    plt.axis('off')
    plt.show()

# Example: Medical test probabilities
plot_conditional_probability(0.01, 0.95, 0.05)
```

## Common Pitfalls

1. **Confusing P(A|B) and P(B|A)**
   - Problem: Mixing up conditional probabilities
   - Solution: Always identify which event is given

2. **Ignoring Prior Probabilities**
   - Problem: Not considering base rates
   - Solution: Always include prior probabilities

3. **Assuming Independence**
   - Problem: Assuming events are independent when they're not
   - Solution: Check for dependencies

## Applications in Machine Learning

### 1. Naive Bayes Classifier
```python
from sklearn.naive_bayes import GaussianNB

def train_naive_bayes(X, y):
    model = GaussianNB()
    model.fit(X, y)
    return model

# Example usage
model = train_naive_bayes(X_train, y_train)
predictions = model.predict(X_test)
```

### 2. Spam Detection
```python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

def train_spam_detector(emails, labels):
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(emails)
    model = MultinomialNB()
    model.fit(X, labels)
    return model, vectorizer
```

### 3. Medical Diagnosis
```python
def calculate_diagnosis_probability(symptoms, disease_probs, symptom_probs):
    # Calculate probability of each disease given symptoms
    total_prob = 0
    for disease, prob in disease_probs.items():
        symptom_prob = 1
        for symptom in symptoms:
            symptom_prob *= symptom_probs[disease][symptom]
        total_prob += prob * symptom_prob
    return total_prob
```

## Interview Questions

1. **Basic Concepts**
   - What's the difference between P(A|B) and P(B|A)?
   - How does Bayes' Theorem help in updating probabilities?
   - When would you use conditional probability?

2. **Practical Applications**
   - How is Bayes' Theorem used in spam detection?
   - What's the role of prior probabilities in machine learning?
   - How do you handle dependent features in Naive Bayes?

## Exercises

1. Calculate conditional probabilities:
   ```python
   # a) Probability of having a disease given a positive test
   # b) Probability of spam given certain words
   # c) Probability of rain given temperature and humidity
   ```

2. Implement a simple Naive Bayes classifier:
   ```python
   def simple_naive_bayes(features, class_probs, feature_probs):
       predictions = []
       for feature_set in features:
           class_scores = []
           for class_label, class_prob in class_probs.items():
               score = class_prob
               for feature, value in feature_set.items():
                   score *= feature_probs[class_label][feature][value]
               class_scores.append((class_label, score))
           predictions.append(max(class_scores, key=lambda x: x[1])[0])
       return predictions
   ```

3. Create a visualization of Bayes' Theorem:
   ```python
   def plot_bayes_update(prior, likelihood, evidence):
       posterior = (likelihood * prior) / evidence
       plt.figure(figsize=(10, 6))
       plt.bar(['Prior', 'Posterior'], [prior, posterior])
       plt.title('Bayesian Update')
       plt.ylim(0, 1)
       plt.show()
   ```

## Additional Resources

- [Bayes' Theorem in Machine Learning](https://www.analyticsvidhya.com/blog/2017/09/naive-bayes-explained/)
- [Conditional Probability Tutorial](https://www.khanacademy.org/math/statistics-probability/probability-library)
- [Bayesian Statistics](https://www.statsmodels.org/stable/index.html) 