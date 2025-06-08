# Hypothesis Testing

## What is Hypothesis Testing?

Hypothesis testing is a statistical method used to make decisions about population parameters based on sample data. It helps us determine if observed differences are statistically significant or just due to chance.

## Why is Hypothesis Testing Important in Machine Learning?

1. **Model Evaluation**
   - Assessing model performance
   - Comparing different models
   - Validating assumptions

2. **Feature Selection**
   - Identifying significant features
   - Understanding relationships
   - Reducing dimensionality

3. **Experimental Design**
   - A/B testing
   - Treatment effects
   - Causal inference

## How Does Hypothesis Testing Work?

### 1. Basic Concepts

#### Null and Alternative Hypotheses
**What**: 
- Null Hypothesis (H₀): Statement of no effect
- Alternative Hypothesis (H₁): Statement of effect

```python
def formulate_hypotheses():
    # Example: Testing if a new model is better
    h0 = "New model performance = Old model performance"
    h1 = "New model performance > Old model performance"
    return h0, h1
```

#### P-value
**What**: Probability of observing the data if H₀ is true
**Implementation**:
```python
from scipy import stats

def calculate_p_value(test_statistic, distribution='normal'):
    if distribution == 'normal':
        return 1 - stats.norm.cdf(test_statistic)
    elif distribution == 't':
        return 1 - stats.t.cdf(test_statistic, df=len(data)-1)

# Example
test_stat = 2.5
p_value = calculate_p_value(test_stat)  # 0.0062
```

### 2. Common Tests

#### Z-test
**What**: Test for population mean when σ is known
**Formula**: \[ z = \frac{\bar{x} - \mu}{\sigma/\sqrt{n}} \]

```python
def z_test(sample, population_mean, population_std):
    sample_mean = np.mean(sample)
    n = len(sample)
    z_score = (sample_mean - population_mean) / (population_std / np.sqrt(n))
    p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
    return z_score, p_value

# Example
sample = np.random.normal(0, 1, 100)
z_score, p_value = z_test(sample, 0, 1)
```

#### T-test
**What**: Test for population mean when σ is unknown
**Formula**: \[ t = \frac{\bar{x} - \mu}{s/\sqrt{n}} \]

```python
def t_test(sample, population_mean):
    sample_mean = np.mean(sample)
    sample_std = np.std(sample, ddof=1)
    n = len(sample)
    t_score = (sample_mean - population_mean) / (sample_std / np.sqrt(n))
    p_value = 2 * (1 - stats.t.cdf(abs(t_score), df=n-1))
    return t_score, p_value

# Example
sample = np.random.normal(0, 1, 100)
t_score, p_value = t_test(sample, 0)
```

#### Chi-square Test
**What**: Test for categorical data independence
**Formula**: \[ \chi^2 = \sum \frac{(O - E)^2}{E} \]

```python
def chi_square_test(observed, expected):
    chi2_stat = np.sum((observed - expected)**2 / expected)
    p_value = 1 - stats.chi2.cdf(chi2_stat, df=len(observed)-1)
    return chi2_stat, p_value

# Example
observed = np.array([10, 15, 20])
expected = np.array([15, 15, 15])
chi2_stat, p_value = chi_square_test(observed, expected)
```

## Visualizing Hypothesis Tests

```python
import matplotlib.pyplot as plt
import seaborn as sns

def plot_hypothesis_test(test_stat, critical_value, test_type='two-tailed'):
    plt.figure(figsize=(10, 6))
    
    # Plot distribution
    x = np.linspace(-4, 4, 100)
    if test_type == 't':
        y = stats.t.pdf(x, df=len(data)-1)
    else:
        y = stats.norm.pdf(x)
    
    plt.plot(x, y, 'b-', label='Distribution')
    
    # Plot test statistic and critical value
    plt.axvline(test_stat, color='r', linestyle='--', 
                label='Test Statistic')
    plt.axvline(critical_value, color='g', linestyle='--', 
                label='Critical Value')
    
    # Shade rejection region
    if test_type == 'two-tailed':
        plt.fill_between(x, y, where=(x < -critical_value) | (x > critical_value),
                        color='red', alpha=0.3)
    else:
        plt.fill_between(x, y, where=x > critical_value,
                        color='red', alpha=0.3)
    
    plt.title('Hypothesis Test Visualization')
    plt.legend()
    plt.show()

# Example
test_stat = 2.5
critical_value = 1.96
plot_hypothesis_test(test_stat, critical_value)
```

## Common Pitfalls

1. **P-hacking**
   - Problem: Multiple testing without correction
   - Solution: Use Bonferroni correction

2. **Sample Size Issues**
   - Problem: Too small or too large samples
   - Solution: Power analysis

3. **Assumption Violations**
   - Problem: Not checking test assumptions
   - Solution: Verify assumptions before testing

## Applications in Machine Learning

### 1. Model Comparison
```python
def compare_models(model1_scores, model2_scores):
    t_stat, p_value = stats.ttest_rel(model1_scores, model2_scores)
    return t_stat, p_value

# Example
model1_scores = [0.85, 0.82, 0.88, 0.84, 0.86]
model2_scores = [0.83, 0.81, 0.85, 0.82, 0.84]
t_stat, p_value = compare_models(model1_scores, model2_scores)
```

### 2. Feature Selection
```python
def feature_significance(X, y):
    p_values = []
    for feature in X.T:
        t_stat, p_value = stats.ttest_ind(feature[y==0], feature[y==1])
        p_values.append(p_value)
    return np.array(p_values)

# Example
X = np.random.randn(100, 5)
y = np.random.randint(0, 2, 100)
p_values = feature_significance(X, y)
```

### 3. A/B Testing
```python
def ab_test(control_data, treatment_data):
    t_stat, p_value = stats.ttest_ind(control_data, treatment_data)
    return t_stat, p_value

# Example
control = np.random.normal(0, 1, 100)
treatment = np.random.normal(0.5, 1, 100)
t_stat, p_value = ab_test(control, treatment)
```

## Interview Questions

1. **Basic Concepts**
   - What's the difference between Type I and Type II errors?
   - How do you choose between different statistical tests?
   - What's the relationship between p-value and significance level?

2. **Practical Applications**
   - How do you handle multiple hypothesis testing?
   - What's the role of hypothesis testing in model evaluation?
   - How do you determine sample size for a test?

## Exercises

1. Implement different hypothesis tests:
   ```python
   # a) Paired t-test
   # b) ANOVA
   # c) Mann-Whitney U test
   ```

2. Create a function for multiple testing correction:
   ```python
   def bonferroni_correction(p_values, alpha=0.05):
       n_tests = len(p_values)
       corrected_alpha = alpha / n_tests
       return p_values < corrected_alpha
   ```

3. Implement power analysis:
   ```python
   def calculate_power(effect_size, sample_size, alpha=0.05):
       return stats.norm.cdf(effect_size * np.sqrt(sample_size) - 
                           stats.norm.ppf(1 - alpha))
   ```

## Additional Resources

- [SciPy Statistics](https://docs.scipy.org/doc/scipy/reference/stats.html)
- [StatsModels](https://www.statsmodels.org/stable/index.html)
- [Hypothesis Testing in Python](https://www.statsmodels.org/stable/stats.html) 