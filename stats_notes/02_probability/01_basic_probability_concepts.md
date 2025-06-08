# Basic Probability Concepts

## What is Probability?

Probability is a measure of the likelihood that an event will occur. It's a number between 0 and 1, where:
- 0 means the event will never occur
- 1 means the event will always occur
- Values in between represent varying degrees of likelihood

## Why is Probability Important in Machine Learning?

1. **Model Uncertainty**
   - Helps quantify uncertainty in predictions
   - Essential for probabilistic models
   - Used in confidence intervals

2. **Decision Making**
   - Basis for many ML algorithms
   - Helps in risk assessment
   - Used in classification problems

3. **Data Analysis**
   - Understanding data distributions
   - Feature importance
   - Model evaluation

## How Does Probability Work?

### 1. Basic Concepts

#### Sample Space (S)
**What**: The set of all possible outcomes
**Example**: 
```python
# Rolling a die
sample_space = {1, 2, 3, 4, 5, 6}
```

#### Event (E)
**What**: A subset of the sample space
**Example**:
```python
# Event: Rolling an even number
event = {2, 4, 6}
```

#### Probability of an Event
**Formula**: \[ P(E) = \frac{number\ of\ favorable\ outcomes}{total\ number\ of\ possible\ outcomes} \]

```python
def calculate_probability(event, sample_space):
    return len(event) / len(sample_space)

# Example
sample_space = {1, 2, 3, 4, 5, 6}
even_numbers = {2, 4, 6}
prob_even = calculate_probability(even_numbers, sample_space)  # 0.5
```

### 2. Types of Events

#### Independent Events
**What**: Events where the occurrence of one doesn't affect the other
**Formula**: \[ P(A \cap B) = P(A) \times P(B) \]

```python
def independent_probability(prob_a, prob_b):
    return prob_a * prob_b

# Example: Probability of getting heads twice in a row
prob_heads = 0.5
prob_two_heads = independent_probability(prob_heads, prob_heads)  # 0.25
```

#### Dependent Events
**What**: Events where the occurrence of one affects the other
**Formula**: \[ P(A \cap B) = P(A) \times P(B|A) \]

```python
def dependent_probability(prob_a, prob_b_given_a):
    return prob_a * prob_b_given_a

# Example: Drawing two cards without replacement
prob_first_ace = 4/52
prob_second_ace = 3/51
prob_two_aces = dependent_probability(prob_first_ace, prob_second_ace)
```

### 3. Complementary Events
**What**: Events that are mutually exclusive and exhaustive
**Formula**: \[ P(A') = 1 - P(A) \]

```python
def complementary_probability(prob_a):
    return 1 - prob_a

# Example: Probability of not rolling a 6
prob_not_six = complementary_probability(1/6)  # 5/6
```

## Visualizing Probability

```python
import matplotlib.pyplot as plt
import numpy as np

def plot_probability_distribution(probabilities, labels):
    plt.figure(figsize=(10, 6))
    plt.bar(labels, probabilities)
    plt.title('Probability Distribution')
    plt.xlabel('Outcomes')
    plt.ylabel('Probability')
    plt.ylim(0, 1)
    plt.show()

# Example: Probability of rolling each number on a die
numbers = [1, 2, 3, 4, 5, 6]
probs = [1/6] * 6
plot_probability_distribution(probs, numbers)
```

## Common Pitfalls

1. **Confusing Independent and Dependent Events**
   - Problem: Assuming events are independent when they're not
   - Solution: Always check if one event affects the other

2. **Ignoring Sample Space**
   - Problem: Not considering all possible outcomes
   - Solution: Define sample space before calculating probabilities

3. **Misinterpreting Probability**
   - Problem: Thinking probability guarantees outcomes
   - Solution: Remember probability is about likelihood, not certainty

## Applications in Machine Learning

### 1. Classification Problems
```python
def calculate_class_probability(features, class_prob, feature_probs):
    # Naive Bayes example
    prob = class_prob
    for feature, feature_prob in zip(features, feature_probs):
        prob *= feature_prob
    return prob
```

### 2. Model Evaluation
```python
def calculate_accuracy(predictions, actual):
    correct = sum(p == a for p, a in zip(predictions, actual))
    return correct / len(predictions)
```

### 3. Feature Selection
```python
def calculate_feature_importance(feature, target):
    # Mutual Information example
    return mutual_info_score(feature, target)
```

## Interview Questions

1. **Basic Concepts**
   - What's the difference between independent and dependent events?
   - How do you calculate conditional probability?
   - What's the relationship between probability and statistics?

2. **Practical Applications**
   - How is probability used in machine learning?
   - How do you handle uncertainty in predictions?
   - What's the role of probability in model evaluation?

## Exercises

1. Calculate the probability of:
   ```python
   # a) Rolling a sum of 7 with two dice
   # b) Drawing two aces from a deck without replacement
   # c) Getting at least one head in three coin tosses
   ```

2. Create a visualization for the probability distribution of:
   ```python
   # Sum of two dice rolls
   sums = range(2, 13)
   probabilities = [1/36, 2/36, 3/36, 4/36, 5/36, 6/36, 5/36, 4/36, 3/36, 2/36, 1/36]
   ```

3. Implement a function to calculate conditional probability:
   ```python
   def conditional_probability(prob_a_and_b, prob_a):
       return prob_a_and_b / prob_a
   ```

## Additional Resources

- [Probability for Data Science](https://www.probabilitycourse.com/)
- [Khan Academy Probability](https://www.khanacademy.org/math/statistics-probability)
- [Statistics and Probability in Python](https://docs.scipy.org/doc/scipy/reference/stats.html) 