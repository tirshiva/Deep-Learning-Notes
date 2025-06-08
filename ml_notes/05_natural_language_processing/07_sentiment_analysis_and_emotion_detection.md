# Sentiment Analysis and Emotion Detection

## Background and Introduction
Sentiment Analysis and Emotion Detection are NLP tasks that involve analyzing text to determine the emotional tone and sentiment expressed. These techniques are crucial for understanding user feedback, social media analysis, and customer experience management.

## What are Sentiment Analysis and Emotion Detection?
Key aspects include:
1. Sentiment classification
2. Emotion recognition
3. Polarity detection
4. Aspect-based analysis
5. Multi-label classification

## Why Sentiment Analysis and Emotion Detection?
1. **Customer Insights**: Understand customer feedback
2. **Social Media Analysis**: Monitor brand sentiment
3. **Market Research**: Analyze product reviews
4. **User Experience**: Improve customer service
5. **Content Analysis**: Understand audience response

## How to Implement Sentiment Analysis and Emotion Detection?

### 1. Basic Sentiment Analysis
```python
import tensorflow as tf
from transformers import BertForSequenceClassification, BertTokenizer
import numpy as np

class SentimentAnalyzer:
    def __init__(self, model_type='bert'):
        self.model_type = model_type
        if model_type == 'bert':
            self.model = self._load_bert_model()
        else:
            self.model = self._build_custom_model()
    
    def _load_bert_model(self):
        # Load pre-trained model
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertForSequenceClassification.from_pretrained(
            'bert-base-uncased',
            num_labels=3  # Positive, Negative, Neutral
        )
        return self.model
    
    def _build_custom_model(self):
        # Create custom sentiment model
        model = tf.keras.Sequential([
            tf.keras.layers.Embedding(10000, 256),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128)),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(3, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def analyze_sentiment(self, text):
        if self.model_type == 'bert':
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
            
            # Get sentiment label
            sentiment_map = {0: 'negative', 1: 'neutral', 2: 'positive'}
            sentiment = sentiment_map[predictions.argmax().item()]
            
            return {
                'sentiment': sentiment,
                'confidence': predictions.max().item()
            }
        else:
            # Implement custom sentiment analysis
            pass
```

### 2. Emotion Detection
```python
from transformers import RobertaForSequenceClassification, RobertaTokenizer
import tensorflow as tf
import numpy as np

class EmotionDetector:
    def __init__(self, model_type='roberta'):
        self.model_type = model_type
        if model_type == 'roberta':
            self.model = self._load_roberta_model()
        else:
            self.model = self._build_custom_model()
    
    def _load_roberta_model(self):
        # Load pre-trained model
        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        self.model = RobertaForSequenceClassification.from_pretrained(
            'roberta-base',
            num_labels=6  # Joy, Sadness, Anger, Fear, Surprise, Neutral
        )
        return self.model
    
    def _build_custom_model(self):
        # Create custom emotion detection model
        model = tf.keras.Sequential([
            tf.keras.layers.Embedding(10000, 256),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128)),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(6, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def detect_emotion(self, text):
        if self.model_type == 'roberta':
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
            
            # Get emotion labels
            emotion_map = {
                0: 'joy',
                1: 'sadness',
                2: 'anger',
                3: 'fear',
                4: 'surprise',
                5: 'neutral'
            }
            
            # Get top emotions
            top_emotions = []
            for i in range(3):  # Get top 3 emotions
                idx = predictions[0].argsort(descending=True)[i]
                emotion = emotion_map[idx.item()]
                confidence = predictions[0][idx].item()
                top_emotions.append({
                    'emotion': emotion,
                    'confidence': confidence
                })
            
            return top_emotions
        else:
            # Implement custom emotion detection
            pass
```

### 3. Advanced Sentiment Analysis
```python
class AdvancedSentimentAnalyzer:
    def __init__(self):
        self.sentiment_model = self._build_sentiment_model()
        self.aspect_model = self._build_aspect_model()
    
    def _build_sentiment_model(self):
        # Create advanced sentiment model
        model = tf.keras.Sequential([
            tf.keras.layers.Embedding(10000, 256),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True)),
            tf.keras.layers.Attention(),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(3, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def _build_aspect_model(self):
        # Create aspect-based sentiment model
        model = tf.keras.Sequential([
            tf.keras.layers.Embedding(10000, 256),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True)),
            tf.keras.layers.Attention(),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(5, activation='softmax')  # 5 aspects
        ])
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def analyze_with_aspects(self, text):
        # Get overall sentiment
        sentiment = self.sentiment_model.predict(text)
        
        # Get aspect-based sentiment
        aspects = self.aspect_model.predict(text)
        
        return {
            'overall_sentiment': sentiment,
            'aspect_sentiment': aspects
        }
```

## Model Evaluation

### 1. Performance Metrics
```python
def evaluate_sentiment_analysis(model, test_data):
    # Calculate accuracy
    correct = 0
    total = len(test_data)
    
    for item in test_data:
        text = item['text']
        true_sentiment = item['sentiment']
        
        prediction = model.analyze_sentiment(text)
        if prediction['sentiment'] == true_sentiment:
            correct += 1
    
    accuracy = correct / total
    
    # Calculate F1 score
    from sklearn.metrics import f1_score
    y_true = [item['sentiment'] for item in test_data]
    y_pred = [model.analyze_sentiment(item['text'])['sentiment'] for item in test_data]
    
    f1 = f1_score(y_true, y_pred, average='weighted')
    
    return {
        'accuracy': accuracy,
        'f1_score': f1
    }

def evaluate_emotion_detection(model, test_data):
    # Calculate emotion accuracy
    correct = 0
    total = len(test_data)
    
    for item in test_data:
        text = item['text']
        true_emotion = item['emotion']
        
        predictions = model.detect_emotion(text)
        if predictions[0]['emotion'] == true_emotion:
            correct += 1
    
    accuracy = correct / total
    
    # Calculate emotion distribution
    emotion_counts = {}
    for item in test_data:
        predictions = model.detect_emotion(item['text'])
        emotion = predictions[0]['emotion']
        emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
    
    return {
        'accuracy': accuracy,
        'emotion_distribution': emotion_counts
    }
```

## Common Interview Questions

1. **Q: What are the main approaches to sentiment analysis?**
   - A: Key approaches include:
     - Rule-based methods
     - Machine learning models
     - Deep learning models
     - Transformer-based models
     - Hybrid approaches

2. **Q: How do you handle sarcasm in sentiment analysis?**
   - A: Solutions include:
     - Context analysis
     - Pattern recognition
     - Advanced NLP techniques
     - Multi-modal analysis
     - Domain-specific training

3. **Q: What are the challenges in emotion detection?**
   - A: Challenges include:
     - Cultural differences
     - Context dependence
     - Mixed emotions
     - Subtle expressions
     - Data quality

## Hands-on Task: Sentiment and Emotion Analysis

### Project: Social Media Analyzer
```python
def social_media_analyzer_project():
    # Initialize systems
    sentiment_analyzer = SentimentAnalyzer(model_type='bert')
    emotion_detector = EmotionDetector(model_type='roberta')
    advanced_analyzer = AdvancedSentimentAnalyzer()
    
    # Test sentiment analysis
    texts = [
        "I love this product! It's amazing!",
        "The service was terrible and slow.",
        "It's okay, nothing special."
    ]
    
    sentiment_results = []
    for text in texts:
        # Analyze sentiment
        sentiment = sentiment_analyzer.analyze_sentiment(text)
        
        # Detect emotions
        emotions = emotion_detector.detect_emotion(text)
        
        # Advanced analysis
        advanced = advanced_analyzer.analyze_with_aspects(text)
        
        sentiment_results.append({
            'text': text,
            'sentiment': sentiment,
            'emotions': emotions,
            'advanced_analysis': advanced
        })
    
    # Evaluate results
    sentiment_metrics = evaluate_sentiment_analysis(sentiment_analyzer, sentiment_results)
    emotion_metrics = evaluate_emotion_detection(emotion_detector, sentiment_results)
    
    return {
        'sentiment_results': sentiment_results,
        'sentiment_metrics': sentiment_metrics,
        'emotion_metrics': emotion_metrics
    }
```

## Next Steps
1. Learn about advanced sentiment analysis
2. Study emotion detection techniques
3. Explore multi-modal analysis
4. Practice with real-world applications
5. Learn about model optimization

## Resources
- [Hugging Face Transformers](https://huggingface.co/transformers/)
- [BERT for Sentiment](https://huggingface.co/bert-base-uncased)
- [RoBERTa](https://huggingface.co/roberta-base)
- [Sentiment Analysis Guide](https://www.tensorflow.org/tutorials/text/text_classification_rnn) 