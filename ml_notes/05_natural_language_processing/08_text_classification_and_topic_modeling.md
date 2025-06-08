# Text Classification and Topic Modeling

## Background and Introduction
Text Classification and Topic Modeling are fundamental NLP tasks that involve categorizing text into predefined classes and discovering abstract topics within a collection of documents. These techniques are essential for document organization, content analysis, and information retrieval.

## What are Text Classification and Topic Modeling?
Key aspects include:
1. Text categorization
2. Topic discovery
3. Document clustering
4. Feature extraction
5. Dimensionality reduction

## Why Text Classification and Topic Modeling?
1. **Document Organization**: Categorize and organize documents
2. **Content Analysis**: Understand document themes
3. **Information Retrieval**: Improve search capabilities
4. **Knowledge Discovery**: Extract insights from text
5. **Automation**: Automate document processing

## How to Implement Text Classification and Topic Modeling?

### 1. Text Classification
```python
import tensorflow as tf
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

class TextClassifier:
    def __init__(self, model_type='transformer'):
        self.model_type = model_type
        if model_type == 'transformer':
            self.model = self._load_transformer_model()
        else:
            self.model = self._build_custom_model()
    
    def _load_transformer_model(self):
        # Load pre-trained model
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.model = DistilBertForSequenceClassification.from_pretrained(
            'distilbert-base-uncased',
            num_labels=10  # Number of classes
        )
        return self.model
    
    def _build_custom_model(self):
        # Create custom text classification model
        model = tf.keras.Sequential([
            tf.keras.layers.Embedding(10000, 256),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128)),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(10, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def classify_text(self, text):
        if self.model_type == 'transformer':
            # Tokenize input
            inputs = self.tokenizer(
                text,
                return_tensors='pt',
                max_length=512,
                truncation=True,
                padding=True
            )
            
            # Get model predictions
            outputs = self.model(**inputs)
            predictions = outputs.logits.softmax(dim=1)
            
            # Get class label
            class_idx = predictions.argmax().item()
            confidence = predictions.max().item()
            
            return {
                'class': class_idx,
                'confidence': confidence
            }
        else:
            # Implement custom classification
            pass
```

### 2. Topic Modeling
```python
from gensim import corpora, models
import numpy as np
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

class TopicModeler:
    def __init__(self, model_type='lda'):
        self.model_type = model_type
        if model_type == 'lda':
            self.model = self._build_lda_model()
        else:
            self.model = self._build_custom_model()
    
    def _build_lda_model(self):
        # Create LDA model
        self.vectorizer = CountVectorizer(
            max_features=1000,
            stop_words='english'
        )
        
        self.lda = LatentDirichletAllocation(
            n_components=10,  # Number of topics
            max_iter=10,
            learning_method='online',
            random_state=42
        )
        
        return self.lda
    
    def _build_custom_model(self):
        # Create custom topic model
        self.dictionary = corpora.Dictionary()
        self.corpus = []
        
        return None
    
    def fit(self, documents):
        if self.model_type == 'lda':
            # Transform documents
            X = self.vectorizer.fit_transform(documents)
            
            # Fit LDA model
            self.lda.fit(X)
            
            # Get feature names
            self.feature_names = self.vectorizer.get_feature_names_out()
        else:
            # Implement custom topic modeling
            pass
    
    def get_topics(self, n_words=10):
        if self.model_type == 'lda':
            topics = []
            for topic_idx, topic in enumerate(self.lda.components_):
                top_words = [self.feature_names[i] for i in topic.argsort()[:-n_words-1:-1]]
                topics.append({
                    'topic_id': topic_idx,
                    'words': top_words
                })
            return topics
        else:
            # Implement custom topic extraction
            pass
```

### 3. Advanced Text Classification
```python
class AdvancedTextClassifier:
    def __init__(self):
        self.classifier = self._build_classifier()
        self.vectorizer = self._build_vectorizer()
    
    def _build_classifier(self):
        # Create advanced classifier
        model = tf.keras.Sequential([
            tf.keras.layers.Embedding(10000, 256),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True)),
            tf.keras.layers.Attention(),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(10, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def _build_vectorizer(self):
        # Create TF-IDF vectorizer
        return TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2)
        )
    
    def classify_with_features(self, text):
        # Extract features
        features = self.vectorizer.transform([text])
        
        # Get classification
        prediction = self.classifier.predict(features)
        
        return {
            'prediction': prediction,
            'features': features
        }
```

## Model Evaluation

### 1. Performance Metrics
```python
def evaluate_text_classification(model, test_data):
    # Calculate accuracy
    correct = 0
    total = len(test_data)
    
    for item in test_data:
        text = item['text']
        true_class = item['class']
        
        prediction = model.classify_text(text)
        if prediction['class'] == true_class:
            correct += 1
    
    accuracy = correct / total
    
    # Calculate F1 score
    from sklearn.metrics import f1_score
    y_true = [item['class'] for item in test_data]
    y_pred = [model.classify_text(item['text'])['class'] for item in test_data]
    
    f1 = f1_score(y_true, y_pred, average='weighted')
    
    return {
        'accuracy': accuracy,
        'f1_score': f1
    }

def evaluate_topic_modeling(model, test_data):
    # Calculate topic coherence
    from gensim.models.coherencemodel import CoherenceModel
    
    coherence_model = CoherenceModel(
        model=model,
        texts=test_data,
        dictionary=model.dictionary,
        coherence='c_v'
    )
    
    coherence_score = coherence_model.get_coherence()
    
    # Calculate topic diversity
    topics = model.get_topics()
    unique_words = set()
    for topic in topics:
        unique_words.update(topic['words'])
    
    diversity = len(unique_words) / (len(topics) * len(topics[0]['words']))
    
    return {
        'coherence': coherence_score,
        'diversity': diversity
    }
```

## Common Interview Questions

1. **Q: What are the main approaches to text classification?**
   - A: Key approaches include:
     - Rule-based methods
     - Machine learning models
     - Deep learning models
     - Transformer-based models
     - Hybrid approaches

2. **Q: How do you choose the number of topics in topic modeling?**
   - A: Methods include:
     - Perplexity analysis
     - Coherence scores
     - Domain knowledge
     - Cross-validation
     - Visualization

3. **Q: What are the challenges in text classification?**
   - A: Challenges include:
     - Class imbalance
     - Feature selection
     - Text preprocessing
     - Model interpretability
     - Scalability

## Hands-on Task: Document Classification and Topic Analysis

### Project: News Article Analyzer
```python
def news_article_analyzer_project():
    # Initialize systems
    classifier = TextClassifier(model_type='transformer')
    topic_modeler = TopicModeler(model_type='lda')
    advanced_classifier = AdvancedTextClassifier()
    
    # Test text classification
    articles = [
        "Technology news about AI and machine learning...",
        "Sports coverage of the latest match...",
        "Business analysis of market trends..."
    ]
    
    classification_results = []
    for article in articles:
        # Classify article
        classification = classifier.classify_text(article)
        
        # Advanced classification
        advanced = advanced_classifier.classify_with_features(article)
        
        classification_results.append({
            'article': article,
            'classification': classification,
            'advanced_analysis': advanced
        })
    
    # Test topic modeling
    topic_modeler.fit(articles)
    topics = topic_modeler.get_topics()
    
    # Evaluate results
    classification_metrics = evaluate_text_classification(classifier, classification_results)
    topic_metrics = evaluate_topic_modeling(topic_modeler, articles)
    
    return {
        'classification_results': classification_results,
        'topics': topics,
        'classification_metrics': classification_metrics,
        'topic_metrics': topic_metrics
    }
```

## Next Steps
1. Learn about advanced classification techniques
2. Study topic modeling methods
3. Explore multi-label classification
4. Practice with real-world applications
5. Learn about model optimization

## Resources
- [Hugging Face Transformers](https://huggingface.co/transformers/)
- [DistilBERT](https://huggingface.co/distilbert-base-uncased)
- [Gensim](https://radimrehurek.com/gensim/)
- [Text Classification Guide](https://www.tensorflow.org/tutorials/text/text_classification_rnn) 