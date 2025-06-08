# Bayesian Statistics

## What is Bayesian Statistics?

Bayesian statistics is a statistical approach that uses probability to quantify uncertainty in statistical inferences. It's based on Bayes' Theorem and provides a framework for updating beliefs about parameters as new data becomes available.

## Why is Bayesian Statistics Important in Machine Learning?

1. **Uncertainty Quantification**
   - Probabilistic predictions
   - Confidence intervals
   - Model uncertainty

2. **Prior Knowledge Integration**
   - Incorporating domain expertise
   - Regularization through priors
   - Transfer learning

3. **Model Selection**
   - Bayesian model comparison
   - Occam's razor
   - Evidence-based decisions

## How Does Bayesian Statistics Work?

### 1. Bayes' Theorem

#### What is Bayes' Theorem?
**Definition**: Updates probability of hypothesis given evidence
**Formula**: \[ P(H|E) = \frac{P(E|H)P(H)}{P(E)} \]

```python
def bayes_theorem(prior, likelihood, evidence):
    return (likelihood * prior) / evidence

# Example
prior = 0.3  # P(H)
likelihood = 0.8  # P(E|H)
evidence = 0.5  # P(E)
posterior = bayes_theorem(prior, likelihood, evidence)
```

### 2. Bayesian Inference

#### What is Bayesian Inference?
**Definition**: Process of updating beliefs using Bayes' Theorem
**Components**:
- Prior distribution
- Likelihood function
- Posterior distribution

```python
def bayesian_inference(prior, likelihood, data):
    # Calculate posterior
    posterior = prior * likelihood(data)
    
    # Normalize
    posterior = posterior / np.sum(posterior)
    
    return posterior

# Example
def normal_prior(x, mu=0, sigma=1):
    return np.exp(-(x - mu)**2 / (2 * sigma**2)) / (sigma * np.sqrt(2 * np.pi))

def normal_likelihood(data, mu, sigma=1):
    return np.prod(np.exp(-(data - mu)**2 / (2 * sigma**2)) / 
                  (sigma * np.sqrt(2 * np.pi)))

# Generate data
data = np.random.normal(1, 1, 100)
x = np.linspace(-3, 3, 100)
prior = normal_prior(x)
likelihood = normal_likelihood(data, x)
posterior = bayesian_inference(prior, likelihood, data)
```

### 3. Markov Chain Monte Carlo (MCMC)

#### What is MCMC?
**Definition**: Method for sampling from probability distributions
**Types**:
- Metropolis-Hastings
- Gibbs sampling
- Hamiltonian Monte Carlo

```python
def metropolis_hastings(log_posterior, initial_state, n_steps=1000):
    current_state = initial_state
    samples = [current_state]
    
    for _ in range(n_steps):
        # Propose new state
        proposal = current_state + np.random.normal(0, 0.1)
        
        # Calculate acceptance ratio
        ratio = np.exp(log_posterior(proposal) - log_posterior(current_state))
        
        # Accept or reject
        if np.random.random() < ratio:
            current_state = proposal
        
        samples.append(current_state)
    
    return np.array(samples)

# Example
def log_posterior(x):
    return -0.5 * x**2  # Log of standard normal

samples = metropolis_hastings(log_posterior, initial_state=0)
```

## Visualizing Bayesian Analysis

```python
import matplotlib.pyplot as plt
import seaborn as sns

def plot_bayesian_update(prior, likelihood, posterior, title='Bayesian Update'):
    plt.figure(figsize=(12, 4))
    
    plt.subplot(131)
    plt.plot(x, prior)
    plt.title('Prior')
    
    plt.subplot(132)
    plt.plot(x, likelihood)
    plt.title('Likelihood')
    
    plt.subplot(133)
    plt.plot(x, posterior)
    plt.title('Posterior')
    
    plt.tight_layout()
    plt.show()

def plot_mcmc_trace(samples, title='MCMC Trace'):
    plt.figure(figsize=(10, 6))
    plt.plot(samples)
    plt.title(title)
    plt.xlabel('Step')
    plt.ylabel('Parameter Value')
    plt.show()

def plot_posterior_distribution(samples, title='Posterior Distribution'):
    plt.figure(figsize=(10, 6))
    sns.histplot(samples, kde=True)
    plt.title(title)
    plt.xlabel('Parameter Value')
    plt.ylabel('Density')
    plt.show()

# Example
x = np.linspace(-3, 3, 100)
prior = normal_prior(x)
likelihood = normal_likelihood(data, x)
posterior = bayesian_inference(prior, likelihood, data)
plot_bayesian_update(prior, likelihood, posterior)
```

## Common Pitfalls

1. **Prior Selection**
   - Problem: Choosing inappropriate priors
   - Solution: Sensitivity analysis

2. **Computational Complexity**
   - Problem: Slow convergence
   - Solution: Efficient sampling methods

3. **Model Specification**
   - Problem: Incorrect likelihood
   - Solution: Model checking

## Applications in Machine Learning

### 1. Bayesian Linear Regression
```python
def bayesian_linear_regression(X, y, prior_precision=1.0):
    n_samples, n_features = X.shape
    
    # Prior
    prior_mean = np.zeros(n_features)
    prior_cov = np.eye(n_features) / prior_precision
    
    # Posterior
    posterior_cov = np.linalg.inv(X.T @ X + prior_precision * np.eye(n_features))
    posterior_mean = posterior_cov @ (X.T @ y)
    
    return posterior_mean, posterior_cov

# Example
X = np.random.randn(100, 2)
y = np.random.randn(100)
posterior_mean, posterior_cov = bayesian_linear_regression(X, y)
```

### 2. Bayesian Model Selection
```python
def bayesian_model_selection(models, data):
    # Calculate evidence for each model
    evidences = []
    for model in models:
        # Approximate evidence using MCMC
        samples = metropolis_hastings(model.log_posterior, initial_state=0)
        evidence = np.mean(np.exp(model.log_likelihood(data, samples)))
        evidences.append(evidence)
    
    return np.array(evidences)

# Example
class Model:
    def __init__(self, complexity):
        self.complexity = complexity
    
    def log_posterior(self, x):
        return -0.5 * x**2 - self.complexity
    
    def log_likelihood(self, data, x):
        return -0.5 * np.sum((data - x)**2)

models = [Model(i) for i in range(3)]
data = np.random.normal(0, 1, 100)
evidences = bayesian_model_selection(models, data)
```

### 3. Bayesian Neural Networks
```python
def bayesian_neural_network(X, y, n_hidden=10, n_samples=1000):
    n_samples, n_features = X.shape
    
    # Prior
    weights_prior = np.random.normal(0, 1, (n_features, n_hidden))
    bias_prior = np.random.normal(0, 1, n_hidden)
    
    # MCMC sampling
    samples = []
    for _ in range(n_samples):
        # Propose new weights
        weights = weights_prior + np.random.normal(0, 0.1, weights_prior.shape)
        bias = bias_prior + np.random.normal(0, 0.1, bias_prior.shape)
        
        # Calculate likelihood
        hidden = np.tanh(X @ weights + bias)
        output = np.random.normal(hidden @ np.ones(n_hidden), 0.1)
        likelihood = np.sum(-0.5 * (y - output)**2)
        
        # Accept or reject
        if np.random.random() < np.exp(likelihood):
            samples.append((weights, bias))
    
    return samples

# Example
X = np.random.randn(100, 2)
y = np.random.randn(100)
samples = bayesian_neural_network(X, y)
```

## Interview Questions

1. **Basic Concepts**
   - What's the difference between frequentist and Bayesian statistics?
   - How do you choose priors in Bayesian analysis?
   - What's the role of MCMC in Bayesian inference?

2. **Practical Applications**
   - How do you handle model comparison in Bayesian statistics?
   - What's the importance of posterior predictive checks?
   - How do you quantify uncertainty in Bayesian models?

## Exercises

1. Implement different MCMC methods:
   ```python
   # a) Gibbs sampling
   # b) Hamiltonian Monte Carlo
   # c) NUTS (No U-Turn Sampler)
   ```

2. Create a function for Bayesian model averaging:
   ```python
   def bayesian_model_averaging(models, data, weights):
       predictions = []
       for model, weight in zip(models, weights):
           pred = model.predict(data)
           predictions.append(pred * weight)
       return np.sum(predictions, axis=0)
   ```

3. Implement a function for posterior predictive checks:
   ```python
   def posterior_predictive_check(model, data, n_samples=1000):
       samples = model.sample_posterior(n_samples)
       predictions = [model.predict(sample, data) for sample in samples]
       return np.mean(predictions, axis=0), np.std(predictions, axis=0)
   ```

## Additional Resources

- [PyMC3 Documentation](https://docs.pymc.io/)
- [Stan User Guide](https://mc-stan.org/docs/2_28/stan-users-guide/index.html)
- [Bayesian Methods for Hackers](https://github.com/CamDavidsonPilon/Probabilistic-Programming-and-Bayesian-Methods-for-Hackers) 