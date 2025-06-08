# Self-Organizing Maps (SOMs)

## Background and Introduction
Self-Organizing Maps (SOMs) are a type of artificial neural network that uses unsupervised learning to produce a low-dimensional representation of high-dimensional data. They are particularly useful for visualization and clustering of complex data. SOMs preserve the topological properties of the input space, making them valuable for data exploration and pattern recognition.

## What are Self-Organizing Maps?
SOMs involve:
1. Creating a grid of nodes
2. Learning input patterns
3. Preserving topology
4. Reducing dimensionality
5. Visualizing data structure

## Why Self-Organizing Maps?
1. **Dimensionality Reduction**: Visualize high-dimensional data
2. **Pattern Recognition**: Discover data clusters
3. **Feature Learning**: Extract meaningful features
4. **Data Visualization**: Create intuitive maps
5. **Topology Preservation**: Maintain data relationships

## How to Create Self-Organizing Maps?

### 1. Basic SOM Implementation
```python
import numpy as np
import matplotlib.pyplot as plt
from minisom import MiniSom
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_blobs

def create_sample_data(n_samples=1000, n_features=10):
    # Generate sample data
    X, _ = make_blobs(n_samples=n_samples, n_features=n_features, random_state=42)
    
    return X

def train_som(X, grid_size=(10, 10), sigma=1.0, learning_rate=0.5, n_iterations=1000):
    # Standardize data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Initialize SOM
    som = MiniSom(
        grid_size[0],
        grid_size[1],
        X.shape[1],
        sigma=sigma,
        learning_rate=learning_rate,
        random_seed=42
    )
    
    # Train SOM
    som.train_random(X_scaled, n_iterations)
    
    return {
        'som': som,
        'scaler': scaler,
        'X_scaled': X_scaled
    }

def plot_som_weights(som, grid_size):
    # Plot weight vectors
    plt.figure(figsize=(10, 10))
    plt.pcolor(som.get_weights().T, cmap='viridis')
    plt.colorbar()
    plt.title('SOM Weight Vectors')
    plt.show()

def plot_som_activation(som, X_scaled):
    # Plot activation map
    plt.figure(figsize=(10, 10))
    plt.pcolor(som.distance_map().T, cmap='viridis')
    plt.colorbar()
    plt.title('SOM Activation Map')
    plt.show()

def demonstrate_som():
    # Create sample data
    X = create_sample_data()
    
    # Train SOM
    results = train_som(X)
    
    # Plot results
    plot_som_weights(results['som'], (10, 10))
    plot_som_activation(results['som'], results['X_scaled'])
    
    return results
```

### 2. SOM Clustering
```python
def cluster_som(som, X_scaled):
    # Get cluster assignments
    cluster_assignments = np.array([som.winner(x) for x in X_scaled])
    
    # Create cluster map
    cluster_map = np.zeros((som.get_weights().shape[0], som.get_weights().shape[1]))
    for i, (x, y) in enumerate(cluster_assignments):
        cluster_map[x, y] += 1
    
    return {
        'assignments': cluster_assignments,
        'map': cluster_map
    }

def plot_clusters(cluster_map):
    plt.figure(figsize=(10, 10))
    plt.pcolor(cluster_map.T, cmap='viridis')
    plt.colorbar()
    plt.title('SOM Cluster Map')
    plt.show()

def analyze_clusters(som, X_scaled, cluster_assignments):
    # Calculate cluster statistics
    n_clusters = len(np.unique(cluster_assignments, axis=0))
    cluster_sizes = np.bincount(cluster_assignments[:, 0] * som.get_weights().shape[1] + 
                               cluster_assignments[:, 1])
    
    print(f"Number of clusters: {n_clusters}")
    print("\nCluster sizes:")
    for i, size in enumerate(cluster_sizes):
        if size > 0:
            print(f"Cluster {i}: {size} samples")
    
    return {
        'n_clusters': n_clusters,
        'cluster_sizes': cluster_sizes
    }
```

### 3. SOM Visualization
```python
def visualize_som(som, X_scaled, labels=None):
    # Create figure
    plt.figure(figsize=(15, 5))
    
    # Plot weight vectors
    plt.subplot(131)
    plt.pcolor(som.get_weights().T, cmap='viridis')
    plt.colorbar()
    plt.title('Weight Vectors')
    
    # Plot activation map
    plt.subplot(132)
    plt.pcolor(som.distance_map().T, cmap='viridis')
    plt.colorbar()
    plt.title('Activation Map')
    
    # Plot cluster assignments
    plt.subplot(133)
    cluster_assignments = np.array([som.winner(x) for x in X_scaled])
    cluster_map = np.zeros((som.get_weights().shape[0], som.get_weights().shape[1]))
    for i, (x, y) in enumerate(cluster_assignments):
        cluster_map[x, y] += 1
    plt.pcolor(cluster_map.T, cmap='viridis')
    plt.colorbar()
    plt.title('Cluster Map')
    
    plt.tight_layout()
    plt.show()
```

## Model Evaluation

### 1. Quantization Error
```python
def calculate_quantization_error(som, X_scaled):
    # Calculate quantization error
    error = 0
    for x in X_scaled:
        winner = som.winner(x)
        error += np.linalg.norm(x - som.get_weights()[winner])
    error /= len(X_scaled)
    
    print(f"Quantization Error: {error:.4f}")
    
    return error

def evaluate_som(som, X_scaled):
    # Calculate quantization error
    error = calculate_quantization_error(som, X_scaled)
    
    # Get cluster assignments
    cluster_results = cluster_som(som, X_scaled)
    
    # Analyze clusters
    cluster_stats = analyze_clusters(som, X_scaled, cluster_results['assignments'])
    
    return {
        'quantization_error': error,
        'cluster_stats': cluster_stats
    }
```

### 2. Topology Preservation
```python
def calculate_topology_preservation(som, X_scaled):
    # Calculate topology preservation measure
    preservation = 0
    for i, x in enumerate(X_scaled):
        winner = som.winner(x)
        neighbors = som.get_neighbors(winner)
        for neighbor in neighbors:
            if np.any(np.all(cluster_som(som, X_scaled)['assignments'] == neighbor, axis=1)):
                preservation += 1
    
    preservation /= len(X_scaled) * len(neighbors)
    
    print(f"Topology Preservation: {preservation:.4f}")
    
    return preservation
```

## Common Interview Questions

1. **Q: What is the difference between SOM and K-means clustering?**
   - A: SOM preserves the topological structure of the input data and creates a low-dimensional representation, while K-means only performs clustering without preserving topology. SOM is better for visualization and understanding data structure, while K-means is simpler and faster for pure clustering tasks.

2. **Q: How do you choose the SOM grid size?**
   - A: The grid size should be chosen based on:
     - Data complexity
     - Desired resolution
     - Computational resources
     - Visualization needs
     - Number of expected clusters

3. **Q: What are the advantages and disadvantages of SOM?**
   - A: Advantages:
     - Preserves topology
     - Good for visualization
     - Handles high-dimensional data
     - Reveals data structure
     Disadvantages:
     - Computationally expensive
     - Sensitive to parameters
     - May not scale well
     - Requires careful tuning

## Hands-on Task: Customer Segmentation

### Project: Customer Behavior Analysis
```python
def analyze_customer_behavior():
    # Create sample customer data
    n_customers = 1000
    n_features = 8
    
    # Generate customer features
    X = np.random.randn(n_customers, n_features)
    
    # Train SOM
    results = train_som(X, grid_size=(15, 15))
    
    # Visualize SOM
    visualize_som(results['som'], results['X_scaled'])
    
    # Analyze clusters
    cluster_results = cluster_som(results['som'], results['X_scaled'])
    cluster_stats = analyze_clusters(
        results['som'],
        results['X_scaled'],
        cluster_results['assignments']
    )
    
    # Calculate metrics
    error = calculate_quantization_error(results['som'], results['X_scaled'])
    preservation = calculate_topology_preservation(results['som'], results['X_scaled'])
    
    return {
        'data': X,
        'som_results': results,
        'cluster_stats': cluster_stats,
        'error': error,
        'preservation': preservation
    }
```

## Next Steps
1. Learn about other neural network architectures
2. Study deep learning approaches
3. Explore advanced visualization techniques
4. Practice with real-world datasets
5. Learn about SOM variants and extensions

## Resources
- [MiniSom Documentation](https://github.com/JustGlowing/minisom)
- [SOM Tutorial](https://www.python-course.eu/self-organizing-maps.php)
- [Neural Networks for Machine Learning](https://www.coursera.org/learn/neural-networks)
- [SOM Applications](https://towardsdatascience.com/self-organizing-maps-ff5853a118d4) 