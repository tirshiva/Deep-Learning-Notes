# Dimensionality Reduction

## Background and Introduction
Dimensionality reduction is a crucial technique in machine learning that reduces the number of features in a dataset while preserving important information. It helps combat the curse of dimensionality, improves computational efficiency, and can reveal hidden patterns in data. This technique is essential for visualization, feature extraction, and data compression.

## What is Dimensionality Reduction?
Dimensionality reduction involves:
1. Reducing the number of features
2. Preserving important information
3. Eliminating redundancy
4. Improving computational efficiency
5. Enabling better visualization

## Why Dimensionality Reduction?
1. **Curse of Dimensionality**: Reduces computational complexity
2. **Visualization**: Enables data visualization in 2D/3D
3. **Noise Reduction**: Removes irrelevant features
4. **Feature Extraction**: Discovers meaningful patterns
5. **Storage Efficiency**: Reduces storage requirements

## How to Reduce Dimensionality?

### 1. Principal Component Analysis (PCA)
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris

def perform_pca(data, n_components=None):
    # Standardize data
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    
    # Perform PCA
    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(data_scaled)
    
    # Calculate explained variance
    explained_variance = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance)
    
    return {
        'transformed_data': pca_result,
        'explained_variance': explained_variance,
        'cumulative_variance': cumulative_variance,
        'components': pca.components_,
        'scaler': scaler
    }

def plot_variance_explained(explained_variance, cumulative_variance):
    plt.figure(figsize=(10, 6))
    
    # Plot individual explained variance
    plt.bar(range(1, len(explained_variance) + 1),
            explained_variance,
            alpha=0.5,
            label='Individual')
    
    # Plot cumulative explained variance
    plt.step(range(1, len(cumulative_variance) + 1),
             cumulative_variance,
             where='mid',
             label='Cumulative')
    
    plt.xlabel('Principal Component')
    plt.ylabel('Explained Variance Ratio')
    plt.title('Explained Variance by Principal Components')
    plt.legend()
    plt.show()

def plot_pca_results(pca_result, labels=None):
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(pca_result[:, 0],
                         pca_result[:, 1],
                         c=labels if labels is not None else 'blue',
                         cmap='viridis')
    
    if labels is not None:
        plt.colorbar(scatter)
    
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    plt.title('PCA Results')
    plt.show()

# Example usage
def demonstrate_pca():
    # Load iris dataset
    iris = load_iris()
    X = iris.data
    y = iris.target
    
    # Perform PCA
    pca_results = perform_pca(X)
    
    # Plot variance explained
    plot_variance_explained(
        pca_results['explained_variance'],
        pca_results['cumulative_variance']
    )
    
    # Plot PCA results
    plot_pca_results(pca_results['transformed_data'], y)
    
    return pca_results
```

### 2. t-SNE (t-Distributed Stochastic Neighbor Embedding)
```python
from sklearn.manifold import TSNE

def perform_tsne(data, n_components=2, perplexity=30.0):
    # Perform t-SNE
    tsne = TSNE(
        n_components=n_components,
        perplexity=perplexity,
        random_state=42
    )
    tsne_result = tsne.fit_transform(data)
    
    return tsne_result

def plot_tsne_results(tsne_result, labels=None):
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(tsne_result[:, 0],
                         tsne_result[:, 1],
                         c=labels if labels is not None else 'blue',
                         cmap='viridis')
    
    if labels is not None:
        plt.colorbar(scatter)
    
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.title('t-SNE Results')
    plt.show()

# Example usage
def demonstrate_tsne():
    # Load iris dataset
    iris = load_iris()
    X = iris.data
    y = iris.target
    
    # Perform t-SNE
    tsne_result = perform_tsne(X)
    
    # Plot results
    plot_tsne_results(tsne_result, y)
    
    return tsne_result
```

### 3. UMAP (Uniform Manifold Approximation and Projection)
```python
import umap

def perform_umap(data, n_components=2, n_neighbors=15, min_dist=0.1):
    # Perform UMAP
    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=42
    )
    umap_result = reducer.fit_transform(data)
    
    return umap_result

def plot_umap_results(umap_result, labels=None):
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(umap_result[:, 0],
                         umap_result[:, 1],
                         c=labels if labels is not None else 'blue',
                         cmap='viridis')
    
    if labels is not None:
        plt.colorbar(scatter)
    
    plt.xlabel('UMAP 1')
    plt.ylabel('UMAP 2')
    plt.title('UMAP Results')
    plt.show()

# Example usage
def demonstrate_umap():
    # Load iris dataset
    iris = load_iris()
    X = iris.data
    y = iris.target
    
    # Perform UMAP
    umap_result = perform_umap(X)
    
    # Plot results
    plot_umap_results(umap_result, y)
    
    return umap_result
```

## Model Evaluation

### 1. Reconstruction Error
```python
def calculate_reconstruction_error(original_data, reduced_data, components):
    # Reconstruct data
    reconstructed_data = np.dot(reduced_data, components)
    
    # Calculate mean squared error
    mse = np.mean((original_data - reconstructed_data) ** 2)
    
    return mse

def evaluate_reduction_methods(data):
    # Standardize data
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    
    # Perform different reduction methods
    pca_results = perform_pca(data_scaled, n_components=2)
    tsne_result = perform_tsne(data_scaled)
    umap_result = perform_umap(data_scaled)
    
    # Calculate reconstruction error for PCA
    pca_error = calculate_reconstruction_error(
        data_scaled,
        pca_results['transformed_data'],
        pca_results['components']
    )
    
    print(f'PCA Reconstruction Error: {pca_error:.4f}')
    
    return {
        'pca': pca_results,
        'tsne': tsne_result,
        'umap': umap_result,
        'pca_error': pca_error
    }
```

### 2. Visualization Comparison
```python
def compare_visualizations(data, labels):
    # Standardize data
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    
    # Perform different reduction methods
    pca_results = perform_pca(data_scaled, n_components=2)
    tsne_result = perform_tsne(data_scaled)
    umap_result = perform_umap(data_scaled)
    
    # Create subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    
    # Plot PCA results
    scatter1 = ax1.scatter(pca_results['transformed_data'][:, 0],
                          pca_results['transformed_data'][:, 1],
                          c=labels, cmap='viridis')
    ax1.set_title('PCA')
    plt.colorbar(scatter1, ax=ax1)
    
    # Plot t-SNE results
    scatter2 = ax2.scatter(tsne_result[:, 0],
                          tsne_result[:, 1],
                          c=labels, cmap='viridis')
    ax2.set_title('t-SNE')
    plt.colorbar(scatter2, ax=ax2)
    
    # Plot UMAP results
    scatter3 = ax3.scatter(umap_result[:, 0],
                          umap_result[:, 1],
                          c=labels, cmap='viridis')
    ax3.set_title('UMAP')
    plt.colorbar(scatter3, ax=ax3)
    
    plt.tight_layout()
    plt.show()
```

## Common Interview Questions

1. **Q: What is the difference between PCA and t-SNE?**
   - A: PCA is a linear dimensionality reduction technique that focuses on preserving global structure and variance, while t-SNE is a non-linear technique that preserves local structure and is better for visualization. PCA is deterministic and faster, while t-SNE is stochastic and computationally more expensive.

2. **Q: How do you choose the number of components in PCA?**
   - A: Several methods can be used:
     - Scree plot (elbow method)
     - Cumulative explained variance ratio
     - Kaiser criterion (eigenvalues > 1)
     - Cross-validation
     - Domain knowledge

3. **Q: What are the advantages and disadvantages of UMAP?**
   - A: Advantages:
     - Preserves both local and global structure
     - Faster than t-SNE
     - Works well with high-dimensional data
     Disadvantages:
     - More parameters to tune
     - Less interpretable than PCA
     - May not preserve all relationships

## Hands-on Task: Image Compression

### Project: Image Compression using PCA
```python
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def compress_image(image_path, n_components):
    # Read image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Reshape image
    height, width, channels = image.shape
    image_reshaped = image.reshape(height, width * channels)
    
    # Perform PCA
    pca = PCA(n_components=n_components)
    compressed = pca.fit_transform(image_reshaped)
    
    # Reconstruct image
    reconstructed = pca.inverse_transform(compressed)
    reconstructed = reconstructed.reshape(height, width, channels)
    
    # Clip values to valid range
    reconstructed = np.clip(reconstructed, 0, 255).astype(np.uint8)
    
    # Calculate compression ratio
    original_size = image.nbytes
    compressed_size = compressed.nbytes + pca.components_.nbytes
    compression_ratio = original_size / compressed_size
    
    return {
        'original': image,
        'compressed': reconstructed,
        'compression_ratio': compression_ratio
    }

def demonstrate_compression():
    # Load image
    image_path = 'sample_image.jpg'
    results = compress_image(image_path, n_components=50)
    
    # Display results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    ax1.imshow(results['original'])
    ax1.set_title('Original Image')
    ax1.axis('off')
    
    ax2.imshow(results['compressed'])
    ax2.set_title(f'Compressed Image (Ratio: {results["compression_ratio"]:.2f})')
    ax2.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    return results
```

## Next Steps
1. Learn about other dimensionality reduction techniques
2. Study feature selection methods
3. Explore autoencoders
4. Practice with real-world datasets
5. Learn about manifold learning

## Resources
- [Scikit-learn Dimensionality Reduction](https://scikit-learn.org/stable/modules/decomposition.html)
- [t-SNE Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html)
- [UMAP Documentation](https://umap-learn.readthedocs.io/)
- [Dimensionality Reduction Techniques](https://towardsdatascience.com/dimensionality-reduction-techniques-619aaf6bf664) 