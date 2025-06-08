# Probability Distributions

## What are Probability Distributions?

Probability distributions describe how probabilities are distributed over the values of a random variable. They are fundamental to understanding:
- Data patterns
- Statistical inference
- Machine learning algorithms

## Why are They Important in Machine Learning?

1. **Model Assumptions**
   - Many ML algorithms assume specific distributions
   - Helps in choosing appropriate models
   - Essential for parameter estimation

2. **Data Analysis**
   - Understanding data characteristics
   - Identifying outliers
   - Feature engineering

3. **Predictions**
   - Quantifying uncertainty
   - Making probabilistic predictions
   - Model evaluation

## How Do They Work?

### 1. Discrete Distributions

#### Binomial Distribution
**What**: Distribution of number of successes in n independent trials
**Formula**: \[ P(X = k) = \binom{n}{k} p^k (1-p)^{n-k} \]

```python
from scipy import stats

def binomial_probability(n, k, p):
    return stats.binom.pmf(k, n, p)

# Example: Probability of getting 3 heads in 5 coin tosses
prob = binomial_probability(n=5, k=3, p=0.5)  # 0.3125
```

#### Poisson Distribution
**What**: Distribution of number of events in fixed time/space
**Formula**: \[ P(X = k) = \frac{\lambda^k e^{-\lambda}}{k!} \]

```python
def poisson_probability(k, lambda_param):
    return stats.poisson.pmf(k, lambda_param)

# Example: Probability of 2 customers arriving in an hour
# when average is 3 per hour
prob = poisson_probability(k=2, lambda_param=3)  # 0.224
```

### 2. Continuous Distributions

#### Normal (Gaussian) Distribution
**What**: Bell-shaped distribution of continuous data
**Formula**: \[ f(x) = \frac{1}{\sigma\sqrt{2\pi}} e^{-\frac{(x-\mu)^2}{2\sigma^2}} \]

```python
def normal_probability(x, mu, sigma):
    return stats.norm.pdf(x, mu, sigma)

# Example: Probability density at x=0 for standard normal
prob = normal_probability(x=0, mu=0, sigma=1)  # 0.3989
```

#### Exponential Distribution
**What**: Distribution of time between events
**Formula**: \[ f(x) = \lambda e^{-\lambda x} \]

```python
def exponential_probability(x, lambda_param):
    return stats.expon.pdf(x, scale=1/lambda_param)

# Example: Probability density at x=2 for λ=0.5
prob = exponential_probability(x=2, lambda_param=0.5)  # 0.1839
```

## Visualizing Distributions

```python
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def plot_distribution(dist_type, params, title):
    plt.figure(figsize=(10, 6))
    
    if dist_type == 'normal':
        x = np.linspace(params['mu'] - 4*params['sigma'], 
                       params['mu'] + 4*params['sigma'], 100)
        y = stats.norm.pdf(x, params['mu'], params['sigma'])
    elif dist_type == 'binomial':
        x = np.arange(0, params['n'] + 1)
        y = stats.binom.pmf(x, params['n'], params['p'])
    elif dist_type == 'poisson':
        x = np.arange(0, int(params['lambda'] * 3))
        y = stats.poisson.pmf(x, params['lambda'])
    
    plt.plot(x, y, 'b-', label=f'{dist_type.capitalize()} Distribution')
    plt.title(title)
    plt.xlabel('Value')
    plt.ylabel('Probability')
    plt.legend()
    plt.grid(True)
    plt.show()

# Example: Plot normal distribution
plot_distribution('normal', {'mu': 0, 'sigma': 1}, 
                 'Standard Normal Distribution')
```

## Common Pitfalls

1. **Assuming Normal Distribution**
   - Problem: Assuming all data is normally distributed
   - Solution: Always check distribution shape

2. **Ignoring Parameters**
   - Problem: Not understanding distribution parameters
   - Solution: Study parameter effects on shape

3. **Misinterpreting PDF**
   - Problem: Treating PDF values as probabilities
   - Solution: Remember PDF values can be > 1

## Applications in Machine Learning

### 1. Data Preprocessing
```python
def check_distribution(data):
    # Test for normal distribution
    statistic, p_value = stats.normaltest(data)
    return p_value > 0.05  # True if normal
```

### 2. Feature Engineering
```python
def transform_to_normal(data):
    # Box-Cox transformation
    transformed_data, lambda_param = stats.boxcox(data)
    return transformed_data
```

### 3. Model Selection
```python
def select_model_by_distribution(data):
    if check_distribution(data):
        return "Use models assuming normal distribution"
    else:
        return "Use non-parametric models"
```

## Interview Questions

1. **Basic Concepts**
   - What's the difference between discrete and continuous distributions?
   - When would you use Poisson distribution?
   - How do you check if data follows a normal distribution?

2. **Practical Applications**
   - How do distributions affect model choice?
   - What's the role of distributions in feature engineering?
   - How do you handle non-normal data?

## Exercises

1. Generate and plot different distributions:
   ```python
   # a) Normal distribution with different parameters
   # b) Binomial distribution for different n and p
   # c) Poisson distribution for different λ
   ```

2. Implement a function to check distribution fit:
   ```python
   def check_distribution_fit(data, dist_type):
       if dist_type == 'normal':
           return stats.normaltest(data)
       elif dist_type == 'poisson':
           return stats.kstest(data, 'poisson', 
                             args=(np.mean(data),))
   ```

3. Create a visualization comparing different distributions:
   ```python
   def compare_distributions():
       x = np.linspace(-4, 4, 100)
       plt.plot(x, stats.norm.pdf(x), label='Normal')
       plt.plot(x, stats.t.pdf(x, df=3), label='t-distribution')
       plt.legend()
       plt.show()
   ```

## Additional Resources

- [SciPy Statistics](https://docs.scipy.org/doc/scipy/reference/stats.html)
- [Probability Distributions in Python](https://www.statsmodels.org/stable/index.html)
- [Distribution Fitting](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rv_continuous.fit.html) 