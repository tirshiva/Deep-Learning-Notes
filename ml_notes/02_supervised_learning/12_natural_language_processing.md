# Natural Language Processing (NLP)

## Background and Introduction
Natural Language Processing (NLP) is a branch of machine learning that focuses on enabling computers to understand, interpret, and generate human language. It combines computational linguistics, machine learning, and deep learning to process and analyze large amounts of natural language data. NLP has become increasingly important with the rise of chatbots, virtual assistants, and automated text analysis.

## What is Natural Language Processing?
NLP involves several key tasks:
1. Text Classification
2. Sentiment Analysis
3. Named Entity Recognition
4. Machine Translation
5. Question Answering
6. Text Generation
7. Text Summarization

## Why Natural Language Processing?
1. **Automation**: Automate text-based tasks
2. **Insight Extraction**: Derive insights from text data
3. **Communication**: Enable human-computer interaction
4. **Information Retrieval**: Efficiently search and retrieve information
5. **Content Generation**: Create human-like text

## How to Process Natural Language?

### 1. Text Preprocessing
```python
import nltk
import re
import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import wordnet
import spacy

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

def preprocess_text(text, remove_stopwords=True, lemmatize=True):
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Tokenize
    tokens = word_tokenize(text)
    
    if remove_stopwords:
        # Remove stopwords
        stop_words = set(stopwords.words('english'))
        tokens = [token for token in tokens if token not in stop_words]
    
    if lemmatize:
        # Lemmatize
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    return ' '.join(tokens)

def get_wordnet_pos(word):
    """Map POS tag to WordNet POS tag"""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {
        "J": wordnet.ADJ,
        "N": wordnet.NOUN,
        "V": wordnet.VERB,
        "R": wordnet.ADV
    }
    return tag_dict.get(tag, wordnet.NOUN)

def advanced_preprocessing(text):
    # Load spaCy model
    nlp = spacy.load('en_core_web_sm')
    
    # Process text with spaCy
    doc = nlp(text)
    
    # Extract named entities
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    
    # Extract noun chunks
    noun_chunks = [chunk.text for chunk in doc.noun_chunks]
    
    # Extract POS tags
    pos_tags = [(token.text, token.pos_) for token in doc]
    
    return {
        'entities': entities,
        'noun_chunks': noun_chunks,
        'pos_tags': pos_tags
    }

# Example usage
def demonstrate_preprocessing():
    # Sample text
    text = "The quick brown fox jumps over the lazy dog. It's a beautiful day!"
    
    # Basic preprocessing
    processed_text = preprocess_text(text)
    print("Basic preprocessing:", processed_text)
    
    # Advanced preprocessing
    advanced_results = advanced_preprocessing(text)
    print("\nNamed entities:", advanced_results['entities'])
    print("Noun chunks:", advanced_results['noun_chunks'])
    print("POS tags:", advanced_results['pos_tags'])
    
    return processed_text, advanced_results
```

### 2. Text Vectorization
```python
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from gensim.models import Word2Vec
import numpy as np

def vectorize_text(texts, method='tfidf'):
    if method == 'count':
        # Bag of Words
        vectorizer = CountVectorizer()
        vectors = vectorizer.fit_transform(texts)
        feature_names = vectorizer.get_feature_names_out()
        
    elif method == 'tfidf':
        # TF-IDF
        vectorizer = TfidfVectorizer()
        vectors = vectorizer.fit_transform(texts)
        feature_names = vectorizer.get_feature_names_out()
        
    elif method == 'word2vec':
        # Word2Vec
        tokenized_texts = [text.split() for text in texts]
        model = Word2Vec(tokenized_texts, vector_size=100, window=5, min_count=1)
        vectors = np.array([np.mean([model.wv[word] for word in text.split()], axis=0)
                          for text in texts])
        feature_names = None
    
    return vectors, feature_names

# Example usage
def demonstrate_vectorization():
    # Sample texts
    texts = [
        "The quick brown fox jumps over the lazy dog",
        "A quick brown dog jumps over the lazy fox",
        "The lazy fox sleeps in the sun"
    ]
    
    # Preprocess texts
    processed_texts = [preprocess_text(text) for text in texts]
    
    # Different vectorization methods
    bow_vectors, bow_features = vectorize_text(processed_texts, 'count')
    tfidf_vectors, tfidf_features = vectorize_text(processed_texts, 'tfidf')
    w2v_vectors, _ = vectorize_text(processed_texts, 'word2vec')
    
    return bow_vectors, tfidf_vectors, w2v_vectors
```

### 3. Text Classification
```python
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

def train_text_classifier(texts, labels, method='tfidf'):
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42
    )
    
    # Vectorize texts
    X_train_vec, feature_names = vectorize_text(X_train, method)
    X_test_vec, _ = vectorize_text(X_test, method)
    
    if method in ['count', 'tfidf']:
        # Traditional ML approach
        model = LogisticRegression()
        model.fit(X_train_vec, y_train)
        predictions = model.predict(X_test_vec)
        
    else:
        # Deep Learning approach
        # Tokenize and pad sequences
        tokenizer = Tokenizer(num_words=10000)
        tokenizer.fit_on_texts(X_train)
        X_train_seq = tokenizer.texts_to_sequences(X_train)
        X_test_seq = tokenizer.texts_to_sequences(X_test)
        
        X_train_pad = pad_sequences(X_train_seq, maxlen=100)
        X_test_pad = pad_sequences(X_test_seq, maxlen=100)
        
        # Build model
        model = tf.keras.Sequential([
            tf.keras.layers.Embedding(10000, 100),
            tf.keras.layers.GlobalAveragePooling1D(),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(optimizer='adam',
                     loss='binary_crossentropy',
                     metrics=['accuracy'])
        
        # Train model
        model.fit(X_train_pad, y_train,
                 epochs=10,
                 validation_data=(X_test_pad, y_test))
        
        predictions = (model.predict(X_test_pad) > 0.5).astype(int)
    
    # Evaluate model
    print(classification_report(y_test, predictions))
    
    return model, predictions

# Example usage
def demonstrate_classification():
    # Sample data
    texts = [
        "I love this product, it's amazing!",
        "This is the worst experience ever",
        "The service was okay, nothing special",
        "I'm really happy with my purchase",
        "Terrible customer service, would not recommend"
    ]
    labels = [1, 0, 1, 1, 0]  # 1 for positive, 0 for negative
    
    # Preprocess texts
    processed_texts = [preprocess_text(text) for text in texts]
    
    # Train classifier
    model, predictions = train_text_classifier(processed_texts, labels)
    
    return model, predictions
```

### 4. Sentiment Analysis
```python
from textblob import TextBlob
import vaderSentiment.vaderSentiment as vader

def analyze_sentiment(text, method='vader'):
    if method == 'textblob':
        # TextBlob sentiment analysis
        blob = TextBlob(text)
        sentiment = blob.sentiment
        
        return {
            'polarity': sentiment.polarity,
            'subjectivity': sentiment.subjectivity
        }
    
    elif method == 'vader':
        # VADER sentiment analysis
        analyzer = vader.SentimentIntensityAnalyzer()
        sentiment = analyzer.polarity_scores(text)
        
        return sentiment

# Example usage
def demonstrate_sentiment_analysis():
    # Sample texts
    texts = [
        "I love this product, it's amazing!",
        "This is the worst experience ever",
        "The service was okay, nothing special"
    ]
    
    # Analyze sentiment
    for text in texts:
        print(f"\nText: {text}")
        print("TextBlob:", analyze_sentiment(text, 'textblob'))
        print("VADER:", analyze_sentiment(text, 'vader'))
```

## Model Evaluation

### 1. Text Classification Metrics
```python
from sklearn.metrics import confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt

def evaluate_classifier(y_true, y_pred, y_prob=None):
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()
    
    # ROC curve
    if y_prob is not None:
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2,
                label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        plt.show()
```

### 2. Word Embeddings Visualization
```python
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def visualize_word_embeddings(model, words):
    # Get word vectors
    word_vectors = np.array([model.wv[word] for word in words])
    
    # Reduce dimensionality
    tsne = TSNE(n_components=2, random_state=42)
    word_vectors_2d = tsne.fit_transform(word_vectors)
    
    # Plot
    plt.figure(figsize=(10, 8))
    plt.scatter(word_vectors_2d[:, 0], word_vectors_2d[:, 1])
    
    for i, word in enumerate(words):
        plt.annotate(word, xy=(word_vectors_2d[i, 0], word_vectors_2d[i, 1]))
    
    plt.title('Word Embeddings Visualization')
    plt.show()
```

## Common Interview Questions

1. **Q: What is the difference between stemming and lemmatization?**
   - A: Stemming reduces words to their root form by removing suffixes, which can sometimes result in non-existent words. Lemmatization, on the other hand, reduces words to their base form (lemma) using vocabulary and morphological analysis, ensuring the result is a valid word.

2. **Q: What are word embeddings and why are they important?**
   - A: Word embeddings are dense vector representations of words in a continuous vector space. They capture semantic relationships between words and are important because they allow machines to understand word meanings and relationships, enabling better performance in NLP tasks.

3. **Q: How do you handle out-of-vocabulary words in NLP?**
   - A: Several approaches can be used:
     - Character-level models
     - Subword tokenization (BPE, WordPiece)
     - FastText's character n-grams
     - Using pre-trained embeddings
     - Special tokens for unknown words

## Hands-on Task: Text Classification Project

### Project: News Article Classification
```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

# Create sample news data
np.random.seed(42)
n_samples = 1000
categories = ['sports', 'technology', 'politics', 'entertainment']

# Generate sample texts
texts = []
labels = []

for category in categories:
    for _ in range(n_samples // len(categories)):
        # Generate random text (in practice, you'd use real news articles)
        text = f"This is a sample {category} article. " * 5
        texts.append(text)
        labels.append(category)

# Create DataFrame
data = pd.DataFrame({
    'text': texts,
    'category': labels
})

# Preprocess texts
data['processed_text'] = data['text'].apply(preprocess_text)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    data['processed_text'], data['category'],
    test_size=0.2, random_state=42
)

# Vectorize texts
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train classifier
classifier = LogisticRegression()
classifier.fit(X_train_vec, y_train)

# Make predictions
predictions = classifier.predict(X_test_vec)

# Evaluate model
print(classification_report(y_test, predictions))

# Visualize results
plt.figure(figsize=(10, 6))
pd.Series(y_test).value_counts().plot(kind='bar')
plt.title('Distribution of News Categories')
plt.xlabel('Category')
plt.ylabel('Count')
plt.show()
```

## Next Steps
1. Learn about advanced NLP models (BERT, GPT)
2. Study sequence-to-sequence models
3. Explore attention mechanisms
4. Practice with real-world datasets
5. Learn about multilingual NLP

## Resources
- [NLTK Documentation](https://www.nltk.org/)
- [spaCy Documentation](https://spacy.io/)
- [Hugging Face Transformers](https://huggingface.co/transformers/)
- [Natural Language Processing with Python](https://www.nltk.org/book/) 