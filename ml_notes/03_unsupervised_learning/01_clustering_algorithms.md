# Clustering Algorithms

## Background and Introduction
Clustering is a fundamental unsupervised learning technique that groups similar data points together without prior knowledge of their labels. It's widely used in various domains such as customer segmentation, image compression, and anomaly detection. Clustering helps discover natural groupings in data and provides insights into data structure.

## What is Clustering?
Clustering involves:
1. Grouping similar data points together
2. Identifying natural patterns in data
3. Discovering data structure
4. Reducing data complexity
5. Finding outliers and anomalies

## Why Clustering?
1. **Pattern Discovery**: Find natural groupings in data
2. **Data Reduction**: Simplify complex datasets
3. **Anomaly Detection**: Identify unusual patterns
4. **Feature Learning**: Extract meaningful features
5. **Data Understanding**: Gain insights into data structure

## How to Perform Clustering?

### 1. K-Means Clustering
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

def perform_kmeans(data, n_clusters, random_state=42):
    # Standardize data
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    
    # Perform K-means clustering
    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=10
    )
    clusters = kmeans.fit_predict(data_scaled)
    
    # Calculate metrics
    inertia = kmeans.inertia_
    silhouette = silhouette_score(data_scaled, clusters)
    
    return {
        'clusters': clusters,
        'centers': kmeans.cluster_centers_,
        'inertia': inertia,
        'silhouette': silhouette
    }

def find_optimal_clusters(data, max_clusters=10):
    # Calculate metrics for different numbers of clusters
    inertias = []
    silhouette_scores = []
    
    for n_clusters in range(2, max_clusters + 1):
        kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=42,
            n_init=10
        )
        clusters = kmeans.fit_predict(data)
        
        inertias.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(data, clusters))
    
    # Plot results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    ax1.plot(range(2, max_clusters + 1), inertias, 'bo-')
    ax1.set_xlabel('Number of Clusters')
    ax1.set_ylabel('Inertia')
    ax1.set_title('Elbow Method')
    
    ax2.plot(range(2, max_clusters + 1), silhouette_scores, 'ro-')
    ax2.set_xlabel('Number of Clusters')
    ax2.set_ylabel('Silhouette Score')
    ax2.set_title('Silhouette Analysis')
    
    plt.tight_layout()
    plt.show()
    
    return inertias, silhouette_scores

# Example usage
def demonstrate_kmeans():
    # Generate sample data
    np.random.seed(42)
    n_samples = 300
    centers = 4
    
    X = np.random.randn(n_samples, 2)
    X[0:75] += 3
    X[75:150] += 6
    X[150:225] += 9
    
    # Find optimal number of clusters
    inertias, silhouette_scores = find_optimal_clusters(X)
    
    # Perform clustering with optimal number of clusters
    optimal_clusters = 4
    results = perform_kmeans(X, optimal_clusters)
    
    # Plot results
    plt.figure(figsize=(10, 8))
    plt.scatter(X[:, 0], X[:, 1], c=results['clusters'], cmap='viridis')
    plt.scatter(results['centers'][:, 0], results['centers'][:, 1],
                c='red', marker='x', s=200, linewidths=3)
    plt.title('K-means Clustering Results')
    plt.show()
    
    return results
```

### 2. Hierarchical Clustering
```python
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering

def perform_hierarchical_clustering(data, n_clusters=None):
    # Perform hierarchical clustering
    if n_clusters is None:
        # Create linkage matrix
        linkage_matrix = linkage(data, method='ward')
        
        # Plot dendrogram
        plt.figure(figsize=(10, 7))
        dendrogram(linkage_matrix)
        plt.title('Hierarchical Clustering Dendrogram')
        plt.xlabel('Sample Index')
        plt.ylabel('Distance')
        plt.show()
        
        return linkage_matrix
    
    else:
        # Perform clustering with specified number of clusters
        clustering = AgglomerativeClustering(
            n_clusters=n_clusters,
            linkage='ward'
        )
        clusters = clustering.fit_predict(data)
        
        return clusters

# Example usage
def demonstrate_hierarchical():
    # Generate sample data
    np.random.seed(42)
    n_samples = 300
    X = np.random.randn(n_samples, 2)
    X[0:75] += 3
    X[75:150] += 6
    X[150:225] += 9
    
    # Perform hierarchical clustering
    linkage_matrix = perform_hierarchical_clustering(X)
    
    # Perform clustering with 4 clusters
    clusters = perform_hierarchical_clustering(X, n_clusters=4)
    
    # Plot results
    plt.figure(figsize=(10, 8))
    plt.scatter(X[:, 0], X[:, 1], c=clusters, cmap='viridis')
    plt.title('Hierarchical Clustering Results')
    plt.show()
    
    return clusters
```

### 3. DBSCAN Clustering
```python
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors

def perform_dbscan(data, eps=None, min_samples=5):
    if eps is None:
        # Find optimal eps using k-distance graph
        neighbors = NearestNeighbors(n_neighbors=min_samples)
        neighbors.fit(data)
        distances, _ = neighbors.kneighbors(data)
        distances = np.sort(distances[:, min_samples-1])
        
        # Plot k-distance graph
        plt.figure(figsize=(10, 6))
        plt.plot(distances)
        plt.title('K-distance Graph')
        plt.xlabel('Points')
        plt.ylabel('Distance')
        plt.show()
        
        return distances
    
    else:
        # Perform DBSCAN clustering
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        clusters = dbscan.fit_predict(data)
        
        # Plot results
        plt.figure(figsize=(10, 8))
        plt.scatter(data[:, 0], data[:, 1], c=clusters, cmap='viridis')
        plt.title('DBSCAN Clustering Results')
        plt.show()
        
        return clusters

# Example usage
def demonstrate_dbscan():
    # Generate sample data
    np.random.seed(42)
    n_samples = 300
    X = np.random.randn(n_samples, 2)
    X[0:75] += 3
    X[75:150] += 6
    X[150:225] += 9
    
    # Find optimal eps
    distances = perform_dbscan(X)
    
    # Perform clustering with optimal eps
    eps = 0.5
    clusters = perform_dbscan(X, eps=eps)
    
    return clusters
```

## Model Evaluation

### 1. Clustering Metrics
```python
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

def evaluate_clustering(data, clusters):
    # Calculate metrics
    silhouette = silhouette_score(data, clusters)
    calinski = calinski_harabasz_score(data, clusters)
    davies = davies_bouldin_score(data, clusters)
    
    print(f'Silhouette Score: {silhouette:.3f}')
    print(f'Calinski-Harabasz Score: {calinski:.3f}')
    print(f'Davies-Bouldin Score: {davies:.3f}')
    
    return {
        'silhouette': silhouette,
        'calinski': calinski,
        'davies': davies
    }
```

### 2. Cluster Visualization
```python
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

def visualize_clusters(data, clusters, method='pca'):
    if method == 'pca':
        # Reduce dimensionality using PCA
        pca = PCA(n_components=2)
        data_2d = pca.fit_transform(data)
        
        # Plot results
        plt.figure(figsize=(10, 8))
        plt.scatter(data_2d[:, 0], data_2d[:, 1], c=clusters, cmap='viridis')
        plt.title('PCA Visualization of Clusters')
        plt.show()
        
    elif method == 'tsne':
        # Reduce dimensionality using t-SNE
        tsne = TSNE(n_components=2, random_state=42)
        data_2d = tsne.fit_transform(data)
        
        # Plot results
        plt.figure(figsize=(10, 8))
        plt.scatter(data_2d[:, 0], data_2d[:, 1], c=clusters, cmap='viridis')
        plt.title('t-SNE Visualization of Clusters')
        plt.show()
    
    return data_2d
```

## Common Interview Questions

1. **Q: What is the difference between K-means and hierarchical clustering?**
   - A: K-means is a partitioning method that requires specifying the number of clusters beforehand and assigns each point to exactly one cluster. Hierarchical clustering creates a tree-like structure (dendrogram) that shows the relationships between clusters and doesn't require specifying the number of clusters in advance.

2. **Q: How do you choose the optimal number of clusters?**
   - A: Several methods can be used:
     - Elbow method (plotting inertia vs. number of clusters)
     - Silhouette analysis
     - Gap statistic
     - Domain knowledge
     - Visual inspection of results

3. **Q: What are the advantages and disadvantages of DBSCAN?**
   - A: Advantages:
     - Can find clusters of arbitrary shapes
     - Doesn't require specifying number of clusters
     - Can identify outliers
     Disadvantages:
     - Sensitive to parameter selection
     - Struggles with high-dimensional data
     - Performance can be slow on large datasets

## Hands-on Task: Customer Segmentation

### Project: Retail Customer Segmentation
```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

# Create sample customer data
np.random.seed(42)
n_customers = 1000

# Generate features
data = pd.DataFrame({
    'age': np.random.normal(35, 10, n_customers),
    'income': np.random.normal(50000, 20000, n_customers),
    'spending_score': np.random.normal(50, 20, n_customers),
    'purchase_frequency': np.random.normal(5, 2, n_customers)
})

# Standardize data
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# Find optimal number of clusters
inertias, silhouette_scores = find_optimal_clusters(data_scaled)

# Perform clustering
optimal_clusters = 4
results = perform_kmeans(data_scaled, optimal_clusters)

# Add cluster labels to original data
data['cluster'] = results['clusters']

# Analyze clusters
cluster_analysis = data.groupby('cluster').agg({
    'age': 'mean',
    'income': 'mean',
    'spending_score': 'mean',
    'purchase_frequency': 'mean'
}).round(2)

print("\nCluster Analysis:")
print(cluster_analysis)

# Visualize clusters
plt.figure(figsize=(12, 8))
sns.scatterplot(
    data=data,
    x='income',
    y='spending_score',
    hue='cluster',
    palette='viridis'
)
plt.title('Customer Segments')
plt.show()

# Evaluate clustering
metrics = evaluate_clustering(data_scaled, results['clusters'])

# Visualize clusters in 2D
data_2d = visualize_clusters(data_scaled, results['clusters'], method='pca')
```

## Next Steps
1. Learn about density-based clustering
2. Study spectral clustering
3. Explore fuzzy clustering
4. Practice with real-world datasets
5. Learn about cluster validation techniques

## Resources
- [Scikit-learn Clustering Documentation](https://scikit-learn.org/stable/modules/clustering.html)
- [K-means Clustering Tutorial](https://www.datacamp.com/community/tutorials/k-means-clustering-python)
- [Hierarchical Clustering Tutorial](https://www.datacamp.com/community/tutorials/hierarchical-clustering-python)
- [DBSCAN Clustering Tutorial](https://www.datacamp.com/community/tutorials/dbscan-clustering-python) 