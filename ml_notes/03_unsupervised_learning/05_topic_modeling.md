# Topic Modeling

## Background and Introduction
Topic Modeling is an unsupervised learning technique that discovers abstract topics within a collection of documents. It helps in understanding the main themes, organizing large text corpora, and extracting meaningful insights from unstructured text data. This technique is widely used in text mining, document classification, and information retrieval.

## What is Topic Modeling?
Topic Modeling involves:
1. Discovering hidden topics
2. Extracting key themes
3. Organizing documents
4. Understanding document structure
5. Finding semantic relationships

## Why Topic Modeling?
1. **Document Organization**: Categorize large text collections
2. **Content Analysis**: Understand main themes
3. **Information Retrieval**: Improve search results
4. **Text Mining**: Extract insights from text
5. **Recommendation Systems**: Suggest related content

## How to Model Topics?

### 1. Latent Dirichlet Allocation (LDA)
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def preprocess_text(texts):
    # Initialize lemmatizer and stopwords
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    
    # Preprocess each text
    processed_texts = []
    for text in texts:
        # Tokenize
        tokens = word_tokenize(text.lower())
        
        # Remove stopwords and lemmatize
        tokens = [lemmatizer.lemmatize(token) for token in tokens
                 if token.isalpha() and token not in stop_words]
        
        processed_texts.append(' '.join(tokens))
    
    return processed_texts

def create_sample_data():
    # Create sample documents
    documents = [
        "Machine learning is a subset of artificial intelligence.",
        "Deep learning uses neural networks for complex tasks.",
        "Natural language processing helps computers understand text.",
        "Computer vision enables machines to interpret images.",
        "Reinforcement learning trains agents through rewards.",
        "Supervised learning uses labeled data for training.",
        "Unsupervised learning finds patterns in unlabeled data.",
        "Transfer learning applies knowledge to new domains."
    ]
    
    return documents

def perform_lda(documents, n_topics=3):
    # Preprocess documents
    processed_docs = preprocess_text(documents)
    
    # Create document-term matrix
    vectorizer = CountVectorizer(max_df=0.95, min_df=2)
    doc_term_matrix = vectorizer.fit_transform(processed_docs)
    
    # Fit LDA model
    lda = LatentDirichletAllocation(
        n_components=n_topics,
        random_state=42
    )
    lda.fit(doc_term_matrix)
    
    # Get feature names
    feature_names = vectorizer.get_feature_names_out()
    
    return {
        'model': lda,
        'vectorizer': vectorizer,
        'doc_term_matrix': doc_term_matrix,
        'feature_names': feature_names
    }

def print_topics(model, feature_names, n_top_words=5):
    for topic_idx, topic in enumerate(model.components_):
        top_words = [feature_names[i] for i in topic.argsort()[:-n_top_words-1:-1]]
        print(f"Topic {topic_idx + 1}: {' '.join(top_words)}")

def plot_topic_distribution(model, doc_term_matrix):
    # Get topic distribution for documents
    topic_dist = model.transform(doc_term_matrix)
    
    # Create heatmap
    plt.figure(figsize=(10, 6))
    sns.heatmap(topic_dist, cmap='YlOrRd')
    plt.xlabel('Topic')
    plt.ylabel('Document')
    plt.title('Topic Distribution Across Documents')
    plt.show()

def demonstrate_lda():
    # Create sample data
    documents = create_sample_data()
    
    # Perform LDA
    results = perform_lda(documents)
    
    # Print topics
    print("\nDiscovered Topics:")
    print_topics(results['model'], results['feature_names'])
    
    # Plot topic distribution
    plot_topic_distribution(results['model'], results['doc_term_matrix'])
    
    return results
```

### 2. Non-negative Matrix Factorization (NMF)
```python
from sklearn.decomposition import NMF

def perform_nmf(documents, n_topics=3):
    # Preprocess documents
    processed_docs = preprocess_text(documents)
    
    # Create document-term matrix
    vectorizer = CountVectorizer(max_df=0.95, min_df=2)
    doc_term_matrix = vectorizer.fit_transform(processed_docs)
    
    # Fit NMF model
    nmf = NMF(
        n_components=n_topics,
        random_state=42
    )
    nmf.fit(doc_term_matrix)
    
    # Get feature names
    feature_names = vectorizer.get_feature_names_out()
    
    return {
        'model': nmf,
        'vectorizer': vectorizer,
        'doc_term_matrix': doc_term_matrix,
        'feature_names': feature_names
    }

def compare_models(documents):
    # Perform LDA
    lda_results = perform_lda(documents)
    
    # Perform NMF
    nmf_results = perform_nmf(documents)
    
    # Print topics from both models
    print("\nLDA Topics:")
    print_topics(lda_results['model'], lda_results['feature_names'])
    
    print("\nNMF Topics:")
    print_topics(nmf_results['model'], nmf_results['feature_names'])
    
    return {
        'lda': lda_results,
        'nmf': nmf_results
    }
```

### 3. Topic Coherence
```python
from gensim.models.coherencemodel import CoherenceModel
from gensim.corpora.dictionary import Dictionary
from gensim.models import LdaModel

def calculate_coherence(documents, n_topics_range=range(2, 11)):
    # Preprocess documents
    processed_docs = [word_tokenize(doc.lower()) for doc in documents]
    
    # Create dictionary
    dictionary = Dictionary(processed_docs)
    
    # Create corpus
    corpus = [dictionary.doc2bow(doc) for doc in processed_docs]
    
    # Calculate coherence for different numbers of topics
    coherence_scores = []
    for n_topics in n_topics_range:
        # Train LDA model
        lda_model = LdaModel(
            corpus=corpus,
            num_topics=n_topics,
            id2word=dictionary
        )
        
        # Calculate coherence
        coherence_model = CoherenceModel(
            model=lda_model,
            texts=processed_docs,
            dictionary=dictionary,
            coherence='c_v'
        )
        coherence_scores.append(coherence_model.get_coherence())
    
    return coherence_scores

def plot_coherence_scores(n_topics_range, coherence_scores):
    plt.figure(figsize=(10, 6))
    plt.plot(n_topics_range, coherence_scores, marker='o')
    plt.xlabel('Number of Topics')
    plt.ylabel('Coherence Score')
    plt.title('Topic Coherence vs Number of Topics')
    plt.grid(True)
    plt.show()
```

## Model Evaluation

### 1. Perplexity
```python
def calculate_perplexity(model, doc_term_matrix):
    # Calculate perplexity
    perplexity = model.perplexity(doc_term_matrix)
    
    print(f"Model Perplexity: {perplexity:.2f}")
    
    return perplexity

def evaluate_models(documents):
    # Perform LDA
    lda_results = perform_lda(documents)
    
    # Perform NMF
    nmf_results = perform_nmf(documents)
    
    # Calculate perplexity for LDA
    lda_perplexity = calculate_perplexity(
        lda_results['model'],
        lda_results['doc_term_matrix']
    )
    
    # Calculate coherence scores
    coherence_scores = calculate_coherence(documents)
    
    # Plot coherence scores
    plot_coherence_scores(range(2, 11), coherence_scores)
    
    return {
        'lda_perplexity': lda_perplexity,
        'coherence_scores': coherence_scores
    }
```

### 2. Topic Visualization
```python
import pyLDAvis
import pyLDAvis.sklearn

def visualize_topics(model, doc_term_matrix, vectorizer):
    # Create visualization
    vis_data = pyLDAvis.sklearn.prepare(
        model,
        doc_term_matrix,
        vectorizer
    )
    
    # Display visualization
    pyLDAvis.display(vis_data)
```

## Common Interview Questions

1. **Q: What is the difference between LDA and NMF?**
   - A: LDA is a probabilistic model that assumes documents are mixtures of topics and topics are mixtures of words, while NMF is a matrix factorization technique that decomposes the document-term matrix into non-negative matrices. LDA is better for discovering abstract topics, while NMF often produces more interpretable topics.

2. **Q: How do you choose the number of topics?**
   - A: Several methods can be used:
     - Domain knowledge
     - Coherence scores
     - Perplexity
     - Cross-validation
     - Visualization techniques

3. **Q: What are the advantages and disadvantages of topic modeling?**
   - A: Advantages:
     - Discovers hidden themes
     - Works with unlabeled data
     - Provides document organization
     - Enables content analysis
     Disadvantages:
     - Requires parameter tuning
     - May produce overlapping topics
     - Sensitive to preprocessing
     - Computationally expensive

## Hands-on Task: News Article Analysis

### Project: News Topic Modeling
```python
def analyze_news_articles():
    # Create sample news articles
    articles = [
        "The stock market reached new heights today as tech companies surged.",
        "Climate change is causing more frequent extreme weather events.",
        "New AI technology promises to revolutionize healthcare diagnostics.",
        "Global leaders meet to discuss climate change solutions.",
        "Tech giants announce new investments in renewable energy.",
        "Healthcare providers adopt AI for better patient care.",
        "Stock market volatility increases due to economic uncertainty.",
        "Scientists warn about accelerating climate change impacts."
    ]
    
    # Perform topic modeling
    results = perform_lda(articles, n_topics=3)
    
    # Print discovered topics
    print("\nDiscovered Topics:")
    print_topics(results['model'], results['feature_names'])
    
    # Plot topic distribution
    plot_topic_distribution(results['model'], results['doc_term_matrix'])
    
    # Calculate coherence scores
    coherence_scores = calculate_coherence(articles)
    plot_coherence_scores(range(2, 11), coherence_scores)
    
    # Visualize topics
    visualize_topics(
        results['model'],
        results['doc_term_matrix'],
        results['vectorizer']
    )
    
    return {
        'articles': articles,
        'model_results': results,
        'coherence_scores': coherence_scores
    }
```

## Next Steps
1. Learn about other topic modeling algorithms
2. Study hierarchical topic models
3. Explore dynamic topic models
4. Practice with real-world datasets
5. Learn about topic model evaluation techniques

## Resources
- [Scikit-learn Topic Modeling](https://scikit-learn.org/stable/modules/decomposition.html)
- [Gensim Topic Modeling](https://radimrehurek.com/gensim/models/ldamodel.html)
- [pyLDAvis Documentation](https://pyldavis.readthedocs.io/)
- [Topic Modeling Tutorial](https://towardsdatascience.com/topic-modeling-with-latent-dirichlet-allocation-e7c5f48e7763) 