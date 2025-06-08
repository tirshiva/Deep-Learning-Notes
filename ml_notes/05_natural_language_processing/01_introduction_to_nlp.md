# Introduction to Natural Language Processing

## Background and Introduction
Natural Language Processing (NLP) is a field of artificial intelligence that focuses on the interaction between computers and human language. It enables machines to understand, interpret, and generate human language, making it possible to process and analyze large amounts of natural language data.

## What is NLP?
NLP is characterized by:
1. Text processing and analysis
2. Language understanding
3. Machine translation
4. Sentiment analysis
5. Text generation

## Why NLP?
1. **Text Understanding**: Process and analyze text data
2. **Automation**: Automate language-based tasks
3. **Insights**: Extract valuable information from text
4. **Communication**: Enable human-machine interaction
5. **Applications**: Power various language-based applications

## How to Implement NLP?

### 1. Text Preprocessing
```python
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
import re

def preprocess_text(text):
    # Download required NLTK data
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Tokenize text
    tokens = word_tokenize(text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    
    # Stemming
    stemmer = PorterStemmer()
    stemmed_tokens = [stemmer.stem(token) for token in tokens]
    
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    return {
        'tokens': tokens,
        'stemmed_tokens': stemmed_tokens,
        'lemmatized_tokens': lemmatized_tokens
    }
```

### 2. Text Classification
```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

def create_text_classifier():
    # Create pipeline
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('classifier', MultinomialNB())
    ])
    
    return pipeline

def train_text_classifier(pipeline, texts, labels):
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42
    )
    
    # Train model
    pipeline.fit(X_train, y_train)
    
    # Evaluate model
    y_pred = pipeline.predict(X_test)
    report = classification_report(y_test, y_pred)
    
    return {
        'pipeline': pipeline,
        'X_test': X_test,
        'y_test': y_test,
        'y_pred': y_pred,
        'report': report
    }
```

### 3. Sentiment Analysis
```python
from textblob import TextBlob
import pandas as pd

def analyze_sentiment(texts):
    # Create DataFrame
    df = pd.DataFrame({'text': texts})
    
    # Analyze sentiment
    df['polarity'] = df['text'].apply(lambda x: TextBlob(x).sentiment.polarity)
    df['subjectivity'] = df['text'].apply(lambda x: TextBlob(x).sentiment.subjectivity)
    
    # Categorize sentiment
    df['sentiment'] = df['polarity'].apply(lambda x: 
        'positive' if x > 0 else 'negative' if x < 0 else 'neutral'
    )
    
    return df

def plot_sentiment_distribution(df):
    # Plot sentiment distribution
    plt.figure(figsize=(10, 6))
    df['sentiment'].value_counts().plot(kind='bar')
    plt.title('Sentiment Distribution')
    plt.xlabel('Sentiment')
    plt.ylabel('Count')
    plt.show()
```

## Model Evaluation

### 1. Performance Metrics
```python
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_classifier(y_true, y_pred):
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='weighted'),
        'recall': recall_score(y_true, y_pred, average='weighted'),
        'f1': f1_score(y_true, y_pred, average='weighted')
    }
    
    return metrics

def plot_confusion_matrix(y_true, y_pred, labels):
    # Create confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()
```

## Common Interview Questions

1. **Q: What are the main challenges in NLP?**
   - A: Key challenges include:
     - Ambiguity in language
     - Context understanding
     - Handling multiple languages
     - Sarcasm and irony detection
     - Domain-specific terminology
     - Data quality and quantity

2. **Q: What is the difference between stemming and lemmatization?**
   - A: Stemming and lemmatization are both text normalization techniques, but:
     - Stemming reduces words to their root form by removing suffixes
     - Lemmatization reduces words to their base form using vocabulary and morphological analysis
     - Stemming is faster but less accurate
     - Lemmatization is more accurate but computationally expensive

3. **Q: How do you handle out-of-vocabulary words in NLP?**
   - A: Approaches include:
     - Using subword tokenization
     - Character-level models
     - Pre-trained embeddings
     - Data augmentation
     - Transfer learning
     - Context-based methods

## Hands-on Task: Text Classification

### Project: News Article Classification
```python
def news_classification_project():
    # Load data
    from sklearn.datasets import fetch_20newsgroups
    newsgroups = fetch_20newsgroups(subset='all')
    
    # Preprocess texts
    preprocessed_texts = []
    for text in newsgroups.data:
        processed = preprocess_text(text)
        preprocessed_texts.append(' '.join(processed['lemmatized_tokens']))
    
    # Create and train classifier
    pipeline = create_text_classifier()
    results = train_text_classifier(
        pipeline,
        preprocessed_texts,
        newsgroups.target
    )
    
    # Evaluate classifier
    metrics = evaluate_classifier(results['y_test'], results['y_pred'])
    plot_confusion_matrix(
        results['y_test'],
        results['y_pred'],
        newsgroups.target_names
    )
    
    # Analyze sentiment
    sentiment_df = analyze_sentiment(newsgroups.data)
    plot_sentiment_distribution(sentiment_df)
    
    return {
        'pipeline': results['pipeline'],
        'metrics': metrics,
        'sentiment_analysis': sentiment_df
    }
```

## Next Steps
1. Learn about advanced NLP techniques
2. Study transformer models
3. Explore language models
4. Practice with real-world applications
5. Learn about multilingual NLP

## Resources
- [NLTK Documentation](https://www.nltk.org/)
- [spaCy Documentation](https://spacy.io/)
- [Hugging Face Transformers](https://huggingface.co/transformers/)
- [Stanford NLP](https://nlp.stanford.edu/) 