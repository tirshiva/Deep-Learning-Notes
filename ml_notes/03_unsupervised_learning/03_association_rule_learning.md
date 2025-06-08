# Association Rule Learning

## Background and Introduction
Association Rule Learning is a rule-based machine learning method for discovering interesting relationships between variables in large databases. It's particularly useful in market basket analysis, where it helps identify patterns of co-occurrence between items. This technique is widely used in retail, e-commerce, and recommendation systems.

## What is Association Rule Learning?
Association Rule Learning involves:
1. Finding frequent patterns in data
2. Generating rules from these patterns
3. Evaluating rule quality
4. Identifying interesting relationships
5. Making recommendations

## Why Association Rule Learning?
1. **Pattern Discovery**: Find hidden relationships
2. **Recommendation Systems**: Suggest related items
3. **Market Basket Analysis**: Understand customer behavior
4. **Cross-selling**: Identify product associations
5. **Inventory Management**: Optimize product placement

## How to Learn Association Rules?

### 1. Apriori Algorithm
```python
import numpy as np
import pandas as pd
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
import matplotlib.pyplot as plt
import seaborn as sns

def create_basket_data():
    # Create sample transaction data
    transactions = [
        ['milk', 'bread', 'butter'],
        ['bread', 'diapers'],
        ['milk', 'diapers', 'beer'],
        ['milk', 'bread', 'diapers'],
        ['bread', 'diapers', 'beer']
    ]
    
    # Convert to one-hot encoded DataFrame
    unique_items = list(set([item for transaction in transactions for item in transaction]))
    basket_data = pd.DataFrame(0, index=range(len(transactions)), columns=unique_items)
    
    for i, transaction in enumerate(transactions):
        for item in transaction:
            basket_data.loc[i, item] = 1
    
    return basket_data

def find_frequent_itemsets(data, min_support=0.3):
    # Find frequent itemsets
    frequent_itemsets = apriori(data, min_support=min_support, use_colnames=True)
    
    return frequent_itemsets

def generate_association_rules(frequent_itemsets, min_confidence=0.5):
    # Generate association rules
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
    
    return rules

def plot_support_confidence(rules):
    plt.figure(figsize=(10, 6))
    plt.scatter(rules['support'], rules['confidence'], alpha=0.5)
    plt.xlabel('Support')
    plt.ylabel('Confidence')
    plt.title('Support vs Confidence')
    
    # Add labels for points with high lift
    for i, rule in rules.iterrows():
        if rule['lift'] > 1.5:
            plt.annotate(f"{rule['antecedents']} -> {rule['consequents']}",
                        (rule['support'], rule['confidence']))
    
    plt.show()

def demonstrate_apriori():
    # Create sample data
    basket_data = create_basket_data()
    
    # Find frequent itemsets
    frequent_itemsets = find_frequent_itemsets(basket_data)
    print("\nFrequent Itemsets:")
    print(frequent_itemsets)
    
    # Generate rules
    rules = generate_association_rules(frequent_itemsets)
    print("\nAssociation Rules:")
    print(rules)
    
    # Plot support vs confidence
    plot_support_confidence(rules)
    
    return {
        'frequent_itemsets': frequent_itemsets,
        'rules': rules
    }
```

### 2. FP-Growth Algorithm
```python
from mlxtend.frequent_patterns import fpgrowth

def find_frequent_itemsets_fp(data, min_support=0.3):
    # Find frequent itemsets using FP-Growth
    frequent_itemsets = fpgrowth(data, min_support=min_support, use_colnames=True)
    
    return frequent_itemsets

def compare_algorithms(data):
    # Compare Apriori and FP-Growth
    import time
    
    # Time Apriori
    start_time = time.time()
    apriori_itemsets = find_frequent_itemsets(data)
    apriori_time = time.time() - start_time
    
    # Time FP-Growth
    start_time = time.time()
    fp_itemsets = find_frequent_itemsets_fp(data)
    fp_time = time.time() - start_time
    
    print(f"Apriori time: {apriori_time:.2f} seconds")
    print(f"FP-Growth time: {fp_time:.2f} seconds")
    
    return {
        'apriori': apriori_itemsets,
        'fp_growth': fp_itemsets,
        'apriori_time': apriori_time,
        'fp_time': fp_time
    }
```

### 3. Rule Evaluation Metrics
```python
def evaluate_rules(rules):
    # Calculate additional metrics
    rules['conviction'] = (1 - rules['support'].values) / (1 - rules['confidence'].values)
    rules['leverage'] = rules['support'].values - (rules['antecedent support'].values * rules['consequent support'].values)
    
    # Plot metrics
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Support vs Confidence
    sns.scatterplot(data=rules, x='support', y='confidence', ax=ax1)
    ax1.set_title('Support vs Confidence')
    
    # Lift vs Confidence
    sns.scatterplot(data=rules, x='lift', y='confidence', ax=ax2)
    ax2.set_title('Lift vs Confidence')
    
    # Conviction vs Confidence
    sns.scatterplot(data=rules, x='conviction', y='confidence', ax=ax3)
    ax3.set_title('Conviction vs Confidence')
    
    # Leverage vs Support
    sns.scatterplot(data=rules, x='leverage', y='support', ax=ax4)
    ax4.set_title('Leverage vs Support')
    
    plt.tight_layout()
    plt.show()
    
    return rules
```

## Model Evaluation

### 1. Rule Quality Metrics
```python
def analyze_rule_quality(rules):
    # Calculate rule quality metrics
    quality_metrics = {
        'avg_confidence': rules['confidence'].mean(),
        'avg_lift': rules['lift'].mean(),
        'avg_conviction': rules['conviction'].mean(),
        'avg_leverage': rules['leverage'].mean(),
        'num_rules': len(rules)
    }
    
    print("\nRule Quality Metrics:")
    for metric, value in quality_metrics.items():
        print(f"{metric}: {value:.4f}")
    
    return quality_metrics
```

### 2. Rule Visualization
```python
def visualize_rules(rules):
    # Create network graph of rules
    import networkx as nx
    
    # Create directed graph
    G = nx.DiGraph()
    
    # Add edges for rules with high lift
    for _, rule in rules[rules['lift'] > 1.5].iterrows():
        G.add_edge(str(rule['antecedents']), str(rule['consequents']),
                  weight=rule['lift'])
    
    # Draw graph
    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_color='lightblue',
            node_size=1500, arrowsize=20, font_size=8)
    
    # Add edge labels
    edge_labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    
    plt.title('Association Rules Network')
    plt.show()
```

## Common Interview Questions

1. **Q: What is the difference between Apriori and FP-Growth algorithms?**
   - A: Apriori uses a breadth-first search approach and generates candidate itemsets, while FP-Growth uses a depth-first search and builds a compact tree structure. FP-Growth is generally faster and more memory-efficient than Apriori, especially for large datasets.

2. **Q: How do you choose the minimum support and confidence thresholds?**
   - A: The choice depends on:
     - Dataset size and sparsity
     - Business requirements
     - Domain knowledge
     - Desired number of rules
     - Rule quality requirements

3. **Q: What are the advantages and disadvantages of association rule learning?**
   - A: Advantages:
     - Easy to understand and interpret
     - Can find unexpected patterns
     - Works well with categorical data
     Disadvantages:
     - Computationally expensive
     - May generate too many rules
     - Sensitive to parameter settings

## Hands-on Task: Market Basket Analysis

### Project: Retail Store Analysis
```python
def analyze_retail_data():
    # Create sample retail data
    transactions = [
        ['milk', 'bread', 'butter', 'eggs'],
        ['bread', 'diapers', 'beer'],
        ['milk', 'diapers', 'beer', 'cola'],
        ['milk', 'bread', 'diapers', 'beer'],
        ['bread', 'diapers', 'beer', 'cola'],
        ['milk', 'bread', 'butter', 'diapers'],
        ['bread', 'butter', 'diapers'],
        ['milk', 'bread', 'diapers', 'beer', 'cola']
    ]
    
    # Convert to one-hot encoded DataFrame
    unique_items = list(set([item for transaction in transactions for item in transaction]))
    basket_data = pd.DataFrame(0, index=range(len(transactions)), columns=unique_items)
    
    for i, transaction in enumerate(transactions):
        for item in transaction:
            basket_data.loc[i, item] = 1
    
    # Find frequent itemsets
    frequent_itemsets = find_frequent_itemsets(basket_data, min_support=0.2)
    
    # Generate rules
    rules = generate_association_rules(frequent_itemsets, min_confidence=0.5)
    
    # Evaluate rules
    rules = evaluate_rules(rules)
    
    # Analyze rule quality
    quality_metrics = analyze_rule_quality(rules)
    
    # Visualize rules
    visualize_rules(rules)
    
    return {
        'frequent_itemsets': frequent_itemsets,
        'rules': rules,
        'quality_metrics': quality_metrics
    }
```

## Next Steps
1. Learn about other association rule algorithms
2. Study sequential pattern mining
3. Explore closed itemset mining
4. Practice with real-world datasets
5. Learn about rule pruning techniques

## Resources
- [MLxtend Documentation](http://rasbt.github.io/mlxtend/)
- [Association Rule Mining Tutorial](https://www.kdnuggets.com/2016/04/association-rules-apriori-algorithm-tutorial.html)
- [Market Basket Analysis](https://towardsdatascience.com/market-basket-analysis-using-python-95883c18b2d4)
- [FP-Growth Algorithm](https://www.geeksforgeeks.org/ml-frequent-pattern-growth-algorithm/) 