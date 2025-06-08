# Text Classification and Sentiment Analysis

## Background and Introduction
Text Classification and Sentiment Analysis are fundamental NLP tasks that involve categorizing text into predefined classes and determining the emotional tone or sentiment of text, respectively. These tasks are crucial for understanding and organizing large volumes of text data.

## What are Text Classification and Sentiment Analysis?
Key aspects include:
1. Text categorization
2. Sentiment detection
3. Emotion analysis
4. Topic classification
5. Document classification

## Why Text Classification and Sentiment Analysis?
1. **Content Organization**: Categorize and organize text data
2. **Customer Insights**: Understand customer feedback and opinions
3. **Market Analysis**: Monitor market sentiment
4. **Content Moderation**: Filter inappropriate content
5. **Business Intelligence**: Extract valuable insights from text

## How to Implement Text Classification and Sentiment Analysis?

### 1. Text Classification Models
```python
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

def create_traditional_classifier(method='naive_bayes'):
    # Create pipeline
    if method == 'naive_bayes':
        classifier = MultinomialNB()
    elif method == 'logistic_regression':
        classifier = LogisticRegression()
    elif method == 'random_forest':
        classifier = RandomForestClassifier()
    else:
        raise ValueError(f"Unsupported method: {method}")
    
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('classifier', classifier)
    ])
    
    return pipeline

def create_deep_learning_classifier(vocab_size, embedding_dim, max_length):
    model = models.Sequential([
        layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
        layers.Conv1D(128, 5, activation='relu'),
        layers.GlobalMaxPooling1D(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def train_classifier(pipeline, X_train, y_train, X_test, y_test):
    # Train model
    pipeline.fit(X_train, y_train)
    
    # Evaluate model
    y_pred = pipeline.predict(X_test)
    report = classification_report(y_test, y_pred)
    
    return {
        'pipeline': pipeline,
        'y_pred': y_pred,
        'report': report
    }
```

### 2. Sentiment Analysis Models
```python
from textblob import TextBlob
import transformers
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

class SentimentAnalyzer:
    def __init__(self, method='textblob'):
        self.method = method
        if method == 'transformers':
            self.tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased-finetuned-sst-2-english')
            self.model = AutoModelForSequenceClassification.from_pretrained('distilbert-base-uncased-finetuned-sst-2-english')
    
    def analyze(self, text):
        if self.method == 'textblob':
            return self._analyze_textblob(text)
        elif self.method == 'transformers':
            return self._analyze_transformers(text)
    
    def _analyze_textblob(self, text):
        analysis = TextBlob(text)
        return {
            'polarity': analysis.sentiment.polarity,
            'subjectivity': analysis.sentiment.subjectivity,
            'sentiment': 'positive' if analysis.sentiment.polarity > 0 else 'negative'
        }
    
    def _analyze_transformers(self, text):
        inputs = self.tokenizer(text, return_tensors='pt', truncation=True, padding=True)
        outputs = self.model(**inputs)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        
        return {
            'positive_score': predictions[0][1].item(),
            'negative_score': predictions[0][0].item(),
            'sentiment': 'positive' if predictions[0][1] > predictions[0][0] else 'negative'
        }

def analyze_batch(texts, analyzer):
    results = []
    for text in texts:
        result = analyzer.analyze(text)
        results.append(result)
    return results
```

### 3. Advanced Sentiment Analysis
```python
class AdvancedSentimentAnalyzer:
    def __init__(self):
        self.emotion_model = self._load_emotion_model()
        self.aspect_model = self._load_aspect_model()
    
    def _load_emotion_model(self):
        # Load emotion classification model
        model = models.Sequential([
            layers.Embedding(10000, 100),
            layers.LSTM(64),
            layers.Dense(32, activation='relu'),
            layers.Dense(6, activation='softmax')  # 6 basic emotions
        ])
        return model
    
    def _load_aspect_model(self):
        # Load aspect-based sentiment analysis model
        model = models.Sequential([
            layers.Embedding(10000, 100),
            layers.Bidirectional(layers.LSTM(64)),
            layers.Dense(32, activation='relu'),
            layers.Dense(3, activation='softmax')  # positive, negative, neutral
        ])
        return model
    
    def analyze_emotion(self, text):
        # Analyze emotions in text
        emotions = ['joy', 'sadness', 'anger', 'fear', 'surprise', 'neutral']
        prediction = self.emotion_model.predict(text)
        return emotions[np.argmax(prediction)]
    
    def analyze_aspect(self, text, aspects):
        # Analyze sentiment for specific aspects
        results = {}
        for aspect in aspects:
            prediction = self.aspect_model.predict(text)
            results[aspect] = np.argmax(prediction)
        return results
```

## Model Evaluation

### 1. Performance Metrics
```python
def evaluate_sentiment_analysis(y_true, y_pred):
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='weighted'),
        'recall': recall_score(y_true, y_pred, average='weighted'),
        'f1': f1_score(y_true, y_pred, average='weighted')
    }
    
    return metrics

def plot_sentiment_distribution(sentiments):
    # Plot sentiment distribution
    plt.figure(figsize=(10, 6))
    pd.Series(sentiments).value_counts().plot(kind='bar')
    plt.title('Sentiment Distribution')
    plt.xlabel('Sentiment')
    plt.ylabel('Count')
    plt.show()

def plot_emotion_analysis(emotions):
    # Plot emotion distribution
    plt.figure(figsize=(10, 6))
    pd.Series(emotions).value_counts().plot(kind='bar')
    plt.title('Emotion Distribution')
    plt.xlabel('Emotion')
    plt.ylabel('Count')
    plt.show()
```

## Common Interview Questions

1. **Q: What are the main approaches to text classification?**
   - A: Key approaches include:
     - Traditional ML (Naive Bayes, SVM)
     - Deep Learning (CNN, RNN, Transformers)
     - Rule-based systems
     - Hybrid approaches
     - Transfer learning

2. **Q: How do you handle imbalanced classes in text classification?**
   - A: Solutions include:
     - Resampling techniques
     - Class weights
     - Data augmentation
     - Ensemble methods
     - Cost-sensitive learning

3. **Q: What are the challenges in sentiment analysis?**
   - A: Challenges include:
     - Sarcasm and irony
     - Context dependence
     - Multiple sentiments
     - Domain adaptation
     - Language variations

## Hands-on Task: Sentiment Analysis

### Project: Customer Review Analysis
```python
def customer_review_analysis_project():
    # Load data
    import pandas as pd
    reviews_df = pd.read_csv('customer_reviews.csv')
    
    # Create sentiment analyzer
    analyzer = SentimentAnalyzer(method='transformers')
    
    # Analyze sentiments
    sentiments = []
    for review in reviews_df['text']:
        result = analyzer.analyze(review)
        sentiments.append(result['sentiment'])
    
    # Create advanced analyzer
    advanced_analyzer = AdvancedSentimentAnalyzer()
    
    # Analyze emotions
    emotions = []
    for review in reviews_df['text']:
        emotion = advanced_analyzer.analyze_emotion(review)
        emotions.append(emotion)
    
    # Analyze aspects
    aspects = ['price', 'quality', 'service']
    aspect_sentiments = []
    for review in reviews_df['text']:
        aspect_result = advanced_analyzer.analyze_aspect(review, aspects)
        aspect_sentiments.append(aspect_result)
    
    # Plot results
    plot_sentiment_distribution(sentiments)
    plot_emotion_analysis(emotions)
    
    # Create summary
    summary = {
        'sentiment_distribution': pd.Series(sentiments).value_counts().to_dict(),
        'emotion_distribution': pd.Series(emotions).value_counts().to_dict(),
        'aspect_sentiments': pd.DataFrame(aspect_sentiments).mean().to_dict()
    }
    
    return summary
```

## Next Steps
1. Learn about advanced classification techniques
2. Study aspect-based sentiment analysis
3. Explore multilingual sentiment analysis
4. Practice with real-world applications
5. Learn about model optimization

## Resources
- [Text Classification Guide](https://www.tensorflow.org/tutorials/text/text_classification_rnn)
- [Sentiment Analysis Tutorial](https://www.kaggle.com/c/sentiment-analysis-on-movie-reviews)
- [Aspect-Based Sentiment Analysis](https://github.com/jiangqn/Aspect-Based-Sentiment-Analysis)
- [NLP with PyTorch](https://pytorch.org/tutorials/beginner/text_sentiment_ngrams_tutorial.html) 