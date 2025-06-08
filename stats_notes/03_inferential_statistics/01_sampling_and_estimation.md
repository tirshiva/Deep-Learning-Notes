# Sampling and Estimation

## What are Sampling and Estimation?

### Sampling
Sampling is the process of selecting a subset of individuals from a population to estimate characteristics of the whole population.

### Estimation
Estimation is the process of using sample data to make inferences about population parameters.

## Why are They Important in Machine Learning?

1. **Data Collection**
   - Efficient data gathering
   - Cost-effective analysis
   - Practical implementation

2. **Model Development**
   - Training data selection
   - Validation set creation
   - Test set preparation

3. **Performance Evaluation**
   - Cross-validation
   - Bootstrap methods
   - Confidence intervals

## How Do They Work?

### 1. Sampling Methods

#### Simple Random Sampling
**What**: Each member has an equal chance of being selected
**Implementation**:
```python
import numpy as np

def simple_random_sample(data, sample_size):
    return np.random.choice(data, size=sample_size, replace=False)

# Example
population = np.arange(1000)
sample = simple_random_sample(population, 100)
```

#### Stratified Sampling
**What**: Population divided into strata, then sampled from each
**Implementation**:
```python
def stratified_sample(data, labels, sample_size):
    unique_labels = np.unique(labels)
    samples = []
    for label in unique_labels:
        label_data = data[labels == label]
        label_sample = simple_random_sample(label_data, 
                                          sample_size // len(unique_labels))
        samples.extend(label_sample)
    return np.array(samples)

# Example
data = np.random.randn(1000)
labels = np.random.randint(0, 3, 1000)
stratified_sample = stratified_sample(data, labels, 100)
```

#### Systematic Sampling
**What**: Selecting every kth element from a population
**Implementation**:
```python
def systematic_sample(data, sample_size):
    k = len(data) // sample_size
    return data[::k][:sample_size]

# Example
population = np.arange(1000)
sample = systematic_sample(population, 100)
```

### 2. Estimation Methods

#### Point Estimation
**What**: Single value estimate of a population parameter
**Implementation**:
```python
def point_estimate(sample, estimator='mean'):
    if estimator == 'mean':
        return np.mean(sample)
    elif estimator == 'median':
        return np.median(sample)
    elif estimator == 'std':
        return np.std(sample)

# Example
sample = np.random.normal(0, 1, 100)
mean_estimate = point_estimate(sample, 'mean')
```

#### Interval Estimation
**What**: Range of values that likely contains the parameter
**Implementation**:
```python
from scipy import stats

def confidence_interval(sample, confidence=0.95):
    mean = np.mean(sample)
    std_err = stats.sem(sample)
    ci = stats.t.interval(confidence, len(sample)-1, 
                         loc=mean, scale=std_err)
    return ci

# Example
sample = np.random.normal(0, 1, 100)
ci = confidence_interval(sample)
```

## Visualizing Sampling and Estimation

```python
import matplotlib.pyplot as plt
import seaborn as sns

def plot_sampling_distribution(population, samples, true_mean):
    plt.figure(figsize=(12, 6))
    
    # Population distribution
    plt.subplot(1, 2, 1)
    sns.histplot(population, kde=True)
    plt.axvline(true_mean, color='r', linestyle='--', 
                label='True Mean')
    plt.title('Population Distribution')
    
    # Sampling distribution
    plt.subplot(1, 2, 2)
    sample_means = [np.mean(sample) for sample in samples]
    sns.histplot(sample_means, kde=True)
    plt.axvline(np.mean(sample_means), color='r', 
                linestyle='--', label='Mean of Sample Means')
    plt.title('Sampling Distribution')
    
    plt.tight_layout()
    plt.show()

# Example
population = np.random.normal(0, 1, 10000)
samples = [np.random.choice(population, 100) for _ in range(1000)]
plot_sampling_distribution(population, samples, 0)
```

## Common Pitfalls

1. **Sampling Bias**
   - Problem: Non-representative samples
   - Solution: Use appropriate sampling methods

2. **Small Sample Sizes**
   - Problem: Unreliable estimates
   - Solution: Ensure adequate sample size

3. **Ignoring Population Structure**
   - Problem: Missing important subgroups
   - Solution: Use stratified sampling

## Applications in Machine Learning

### 1. Cross-Validation
```python
from sklearn.model_selection import KFold

def k_fold_cross_validation(X, y, k=5):
    kf = KFold(n_splits=k, shuffle=True)
    scores = []
    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        # Train and evaluate model
        score = evaluate_model(X_train, X_test, y_train, y_test)
        scores.append(score)
    return np.mean(scores), np.std(scores)
```

### 2. Bootstrap Sampling
```python
def bootstrap_sample(data, n_samples):
    n = len(data)
    return np.random.choice(data, size=(n_samples, n), replace=True)

def bootstrap_confidence_interval(data, statistic, n_bootstrap=1000, 
                                confidence=0.95):
    bootstrap_samples = bootstrap_sample(data, n_bootstrap)
    bootstrap_statistics = [statistic(sample) for sample in bootstrap_samples]
    return np.percentile(bootstrap_statistics, 
                        [100*(1-confidence)/2, 100*(1+confidence)/2])
```

### 3. Model Evaluation
```python
def evaluate_model_with_confidence(model, X, y, n_splits=5):
    scores = []
    kf = KFold(n_splits=n_splits, shuffle=True)
    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)
        scores.append(score)
    return np.mean(scores), confidence_interval(scores)
```

## Interview Questions

1. **Basic Concepts**
   - What's the difference between population and sample?
   - How do you choose a sampling method?
   - What's the Central Limit Theorem?

2. **Practical Applications**
   - How do you determine sample size?
   - What's the role of sampling in cross-validation?
   - How do you handle imbalanced data in sampling?

## Exercises

1. Implement different sampling methods:
   ```python
   # a) Cluster sampling
   # b) Quota sampling
   # c) Snowball sampling
   ```

2. Create a function to calculate sample size:
   ```python
   def calculate_sample_size(population_size, confidence_level=0.95, 
                           margin_of_error=0.05):
       z_score = stats.norm.ppf((1 + confidence_level) / 2)
       p = 0.5  # Maximum variability
       q = 1 - p
       e = margin_of_error
       n = (z_score**2 * p * q) / (e**2)
       return int(n)
   ```

3. Implement bootstrap resampling:
   ```python
   def bootstrap_resample(data, n_resamples=1000):
       n = len(data)
       resamples = np.random.choice(data, size=(n_resamples, n), 
                                  replace=True)
       return resamples
   ```

## Additional Resources

- [Sampling Methods in Python](https://docs.scipy.org/doc/scipy/reference/stats.html)
- [Bootstrap Methods](https://www.statsmodels.org/stable/index.html)
- [Cross-Validation Techniques](https://scikit-learn.org/stable/modules/cross_validation.html) 