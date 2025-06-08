# Multivariate Analysis

## What is Multivariate Analysis?

Multivariate analysis is a statistical method that examines relationships between multiple variables simultaneously. It's crucial for understanding complex data structures and patterns in high-dimensional spaces, making it fundamental to many machine learning applications.

## Why is Multivariate Analysis Important in Machine Learning?

1. **Dimensionality Reduction**
   - Feature extraction
   - Data compression
   - Pattern recognition

2. **Pattern Recognition**
   - Cluster analysis
   - Classification
   - Anomaly detection

3. **Data Understanding**
   - Variable relationships
   - Data structure
   - Feature importance

## How Does Multivariate Analysis Work?

### 1. Principal Component Analysis (PCA)

#### What is PCA?
**Definition**: Linear dimensionality reduction technique
**Formula**: \[ X_{reduced} = XW \] where W is the matrix of eigenvectors

```python
def pca(X, n_components=2):
    # Center the data
    X_centered = X - np.mean(X, axis=0)
    
    # Calculate covariance matrix
    cov_matrix = np.cov(X_centered.T)
    
    # Calculate eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    
    # Sort eigenvalues and eigenvectors
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # Project data onto principal components
    X_pca = X_centered @ eigenvectors[:, :n_components]
    
    return X_pca, eigenvalues, eigenvectors

# Example
X = np.random.randn(100, 5)
X_pca, eigenvalues, eigenvectors = pca(X)
```

### 2. Factor Analysis

#### What is Factor Analysis?
**Definition**: Identifies underlying latent variables
**Formula**: \[ X = \Lambda F + \epsilon \]

```python
def factor_analysis(X, n_factors=2):
    from sklearn.decomposition import FactorAnalysis
    
    # Fit factor analysis
    fa = FactorAnalysis(n_components=n_factors)
    X_fa = fa.fit_transform(X)
    
    return X_fa, fa.components_

# Example
X = np.random.randn(100, 5)
X_fa, components = factor_analysis(X)
```

### 3. Canonical Correlation Analysis (CCA)

#### What is CCA?
**Definition**: Finds relationships between two sets of variables
**Formula**: \[ \rho = \max_{a,b} \frac{a^T\Sigma_{xy}b}{\sqrt{a^T\Sigma_{xx}a b^T\Sigma_{yy}b}} \]

```python
def canonical_correlation_analysis(X, Y):
    # Center the data
    X_centered = X - np.mean(X, axis=0)
    Y_centered = Y - np.mean(Y, axis=0)
    
    # Calculate covariance matrices
    cov_xx = np.cov(X_centered.T)
    cov_yy = np.cov(Y_centered.T)
    cov_xy = np.cov(X_centered.T, Y_centered.T)[:X.shape[1], X.shape[1]:]
    
    # Calculate canonical correlations
    inv_cov_xx = np.linalg.inv(cov_xx)
    inv_cov_yy = np.linalg.inv(cov_yy)
    
    # Calculate matrix for eigenvalue decomposition
    matrix = inv_cov_xx @ cov_xy @ inv_cov_yy @ cov_xy.T
    
    # Calculate eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    
    # Sort eigenvalues and eigenvectors
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    return eigenvalues, eigenvectors

# Example
X = np.random.randn(100, 3)
Y = np.random.randn(100, 3)
eigenvalues, eigenvectors = canonical_correlation_analysis(X, Y)
```

## Visualizing Multivariate Analysis

```python
import matplotlib.pyplot as plt
import seaborn as sns

def plot_pca_results(X_pca, eigenvalues, title='PCA Results'):
    plt.figure(figsize=(12, 5))
    
    # Plot PCA projection
    plt.subplot(121)
    plt.scatter(X_pca[:, 0], X_pca[:, 1])
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    plt.title('PCA Projection')
    
    # Plot explained variance
    plt.subplot(122)
    explained_variance = eigenvalues / np.sum(eigenvalues)
    plt.plot(np.cumsum(explained_variance))
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('Explained Variance')
    
    plt.tight_layout()
    plt.show()

def plot_factor_loadings(loadings, feature_names, title='Factor Loadings'):
    plt.figure(figsize=(10, 6))
    sns.heatmap(loadings, 
                xticklabels=['Factor ' + str(i+1) for i in range(loadings.shape[1])],
                yticklabels=feature_names,
                cmap='coolwarm',
                center=0)
    plt.title(title)
    plt.show()

def plot_canonical_correlations(correlations, title='Canonical Correlations'):
    plt.figure(figsize=(8, 6))
    plt.plot(correlations, 'o-')
    plt.xlabel('Canonical Variable Pair')
    plt.ylabel('Correlation')
    plt.title(title)
    plt.show()

# Example
X = np.random.randn(100, 5)
X_pca, eigenvalues, eigenvectors = pca(X)
plot_pca_results(X_pca, eigenvalues)
```

## Common Pitfalls

1. **Dimensionality Choice**
   - Problem: Choosing wrong number of components
   - Solution: Scree plot and explained variance

2. **Data Scaling**
   - Problem: Variables on different scales
   - Solution: Standardization

3. **Interpretation**
   - Problem: Complex relationships
   - Solution: Visualization and domain knowledge

## Applications in Machine Learning

### 1. Feature Extraction
```python
def extract_features(X, method='pca', n_components=2):
    if method == 'pca':
        X_reduced, _, _ = pca(X, n_components)
    elif method == 'fa':
        X_reduced, _ = factor_analysis(X, n_components)
    return X_reduced

# Example
X = np.random.randn(100, 5)
X_reduced = extract_features(X, method='pca')
```

### 2. Dimensionality Reduction
```python
def reduce_dimensions(X, method='pca', n_components=2):
    if method == 'pca':
        X_reduced, eigenvalues, _ = pca(X, n_components)
        return X_reduced, eigenvalues
    elif method == 'fa':
        X_reduced, components = factor_analysis(X, n_components)
        return X_reduced, components

# Example
X = np.random.randn(100, 5)
X_reduced, components = reduce_dimensions(X, method='pca')
```

### 3. Pattern Recognition
```python
def find_patterns(X, method='pca', n_components=2):
    if method == 'pca':
        X_reduced, eigenvalues, eigenvectors = pca(X, n_components)
        return X_reduced, eigenvalues, eigenvectors
    elif method == 'fa':
        X_reduced, components = factor_analysis(X, n_components)
        return X_reduced, components

# Example
X = np.random.randn(100, 5)
X_reduced, eigenvalues, eigenvectors = find_patterns(X, method='pca')
```

## Interview Questions

1. **Basic Concepts**
   - What's the difference between PCA and Factor Analysis?
   - How do you choose the number of components?
   - What's the role of eigenvalues in PCA?

2. **Practical Applications**
   - How do you interpret factor loadings?
   - What's the importance of data scaling?
   - How do you validate dimensionality reduction?

## Exercises

1. Implement different dimensionality reduction methods:
   ```python
   # a) t-SNE
   # b) UMAP
   # c) MDS
   ```

2. Create a function for feature selection:
   ```python
   def select_features(X, y, method='pca', n_components=2):
       if method == 'pca':
           X_reduced, _, _ = pca(X, n_components)
       elif method == 'fa':
           X_reduced, _ = factor_analysis(X, n_components)
       return X_reduced
   ```

3. Implement a function for pattern visualization:
   ```python
   def visualize_patterns(X, method='pca', n_components=2):
       if method == 'pca':
           X_reduced, eigenvalues, _ = pca(X, n_components)
           plot_pca_results(X_reduced, eigenvalues)
       elif method == 'fa':
           X_reduced, components = factor_analysis(X, n_components)
           plot_factor_loadings(components, ['Feature ' + str(i) 
                                           for i in range(X.shape[1])])
   ```

## Additional Resources

- [Scikit-learn Decomposition](https://scikit-learn.org/stable/modules/decomposition.html)
- [StatsModels Multivariate](https://www.statsmodels.org/stable/multivariate.html)
- [Multivariate Analysis in Python](https://www.statsmodels.org/stable/examples/index.html#multivariate) 