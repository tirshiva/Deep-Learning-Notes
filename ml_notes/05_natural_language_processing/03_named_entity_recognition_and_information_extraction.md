# Named Entity Recognition and Information Extraction

## Background and Introduction
Named Entity Recognition (NER) and Information Extraction (IE) are NLP tasks that involve identifying and extracting structured information from unstructured text. NER focuses on identifying named entities (people, organizations, locations, etc.), while IE encompasses a broader range of information extraction tasks.

## What are NER and IE?
Key aspects include:
1. Entity identification
2. Entity classification
3. Relationship extraction
4. Event extraction
5. Attribute extraction

## Why NER and IE?
1. **Information Organization**: Structure unstructured data
2. **Knowledge Base Creation**: Build knowledge graphs
3. **Search Enhancement**: Improve search capabilities
4. **Data Analysis**: Extract insights from text
5. **Automation**: Automate information extraction

## How to Implement NER and IE?

### 1. Named Entity Recognition
```python
import spacy
from spacy.tokens import Doc
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np

class NERModel:
    def __init__(self, model_type='spacy'):
        self.model_type = model_type
        if model_type == 'spacy':
            self.model = spacy.load('en_core_web_sm')
        else:
            self.model = self._build_custom_model()
    
    def _build_custom_model(self):
        model = models.Sequential([
            layers.Embedding(10000, 100),
            layers.Bidirectional(layers.LSTM(64, return_sequences=True)),
            layers.TimeDistributed(layers.Dense(32, activation='relu')),
            layers.Dense(9, activation='softmax')  # 9 entity types
        ])
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def extract_entities(self, text):
        if self.model_type == 'spacy':
            doc = self.model(text)
            entities = []
            for ent in doc.ents:
                entities.append({
                    'text': ent.text,
                    'label': ent.label_,
                    'start': ent.start_char,
                    'end': ent.end_char
                })
            return entities
        else:
            # Custom model prediction
            tokens = text.split()
            predictions = self.model.predict(tokens)
            return self._process_predictions(tokens, predictions)
    
    def _process_predictions(self, tokens, predictions):
        entities = []
        current_entity = None
        
        for token, pred in zip(tokens, predictions):
            label = np.argmax(pred)
            if label != 0:  # 0 is 'O' (Outside) tag
                if current_entity is None:
                    current_entity = {
                        'text': token,
                        'label': self.id2label[label],
                        'start': len(' '.join(tokens[:tokens.index(token)])),
                        'end': len(' '.join(tokens[:tokens.index(token)+1]))
                    }
                else:
                    current_entity['text'] += ' ' + token
                    current_entity['end'] = len(' '.join(tokens[:tokens.index(token)+1]))
            else:
                if current_entity is not None:
                    entities.append(current_entity)
                    current_entity = None
        
        return entities
```

### 2. Information Extraction
```python
class InformationExtractor:
    def __init__(self):
        self.ner_model = NERModel()
        self.relation_model = self._build_relation_model()
    
    def _build_relation_model(self):
        model = models.Sequential([
            layers.Embedding(10000, 100),
            layers.Bidirectional(layers.LSTM(64)),
            layers.Dense(32, activation='relu'),
            layers.Dense(10, activation='softmax')  # 10 relation types
        ])
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def extract_relations(self, text):
        # Extract entities
        entities = self.ner_model.extract_entities(text)
        
        # Extract relations between entities
        relations = []
        for i, ent1 in enumerate(entities):
            for j, ent2 in enumerate(entities[i+1:], i+1):
                # Get context between entities
                context = self._get_context(text, ent1, ent2)
                
                # Predict relation
                relation = self._predict_relation(context)
                if relation != 'NO_RELATION':
                    relations.append({
                        'entity1': ent1,
                        'entity2': ent2,
                        'relation': relation
                    })
        
        return relations
    
    def _get_context(self, text, ent1, ent2):
        # Extract text between entities
        start = min(ent1['start'], ent2['start'])
        end = max(ent1['end'], ent2['end'])
        return text[start:end]
    
    def _predict_relation(self, context):
        # Predict relation type
        prediction = self.relation_model.predict([context])
        return self.id2relation[np.argmax(prediction)]
```

### 3. Event Extraction
```python
class EventExtractor:
    def __init__(self):
        self.ner_model = NERModel()
        self.event_model = self._build_event_model()
    
    def _build_event_model(self):
        model = models.Sequential([
            layers.Embedding(10000, 100),
            layers.Bidirectional(layers.LSTM(64)),
            layers.Dense(32, activation='relu'),
            layers.Dense(5, activation='softmax')  # 5 event types
        ])
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def extract_events(self, text):
        # Extract entities
        entities = self.ner_model.extract_entities(text)
        
        # Extract events
        events = []
        sentences = text.split('.')
        
        for sentence in sentences:
            # Predict event type
            event_type = self._predict_event_type(sentence)
            
            if event_type != 'NO_EVENT':
                # Extract event arguments
                arguments = self._extract_event_arguments(sentence, entities)
                
                events.append({
                    'type': event_type,
                    'text': sentence,
                    'arguments': arguments
                })
        
        return events
    
    def _predict_event_type(self, sentence):
        # Predict event type
        prediction = self.event_model.predict([sentence])
        return self.id2event[np.argmax(prediction)]
    
    def _extract_event_arguments(self, sentence, entities):
        # Extract event arguments from entities
        arguments = []
        for entity in entities:
            if entity['text'] in sentence:
                arguments.append(entity)
        return arguments
```

## Model Evaluation

### 1. Performance Metrics
```python
def evaluate_ner(y_true, y_pred):
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='weighted'),
        'recall': recall_score(y_true, y_pred, average='weighted'),
        'f1': f1_score(y_true, y_pred, average='weighted')
    }
    
    return metrics

def evaluate_relations(y_true, y_pred):
    # Calculate metrics for relation extraction
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='weighted'),
        'recall': recall_score(y_true, y_pred, average='weighted'),
        'f1': f1_score(y_true, y_pred, average='weighted')
    }
    
    return metrics

def plot_entity_distribution(entities):
    # Plot entity type distribution
    plt.figure(figsize=(10, 6))
    pd.Series([ent['label'] for ent in entities]).value_counts().plot(kind='bar')
    plt.title('Entity Type Distribution')
    plt.xlabel('Entity Type')
    plt.ylabel('Count')
    plt.show()
```

## Common Interview Questions

1. **Q: What are the main challenges in NER?**
   - A: Key challenges include:
     - Entity ambiguity
     - Named entity variations
     - Domain adaptation
     - Multilingual NER
     - Entity boundaries
     - Rare entities

2. **Q: How do you handle nested entities in NER?**
   - A: Approaches include:
     - Hierarchical models
     - Multi-task learning
     - Span-based models
     - Recursive models
     - Boundary detection

3. **Q: What are the different types of information extraction?**
   - A: Types include:
     - Named entity recognition
     - Relation extraction
     - Event extraction
     - Attribute extraction
     - Template filling
     - Knowledge base population

## Hands-on Task: Information Extraction

### Project: News Article Analysis
```python
def news_article_analysis_project():
    # Load data
    import pandas as pd
    articles_df = pd.read_csv('news_articles.csv')
    
    # Initialize models
    ner_model = NERModel()
    ie_model = InformationExtractor()
    event_model = EventExtractor()
    
    # Process articles
    results = []
    for article in articles_df['text']:
        # Extract entities
        entities = ner_model.extract_entities(article)
        
        # Extract relations
        relations = ie_model.extract_relations(article)
        
        # Extract events
        events = event_model.extract_events(article)
        
        results.append({
            'entities': entities,
            'relations': relations,
            'events': events
        })
    
    # Analyze results
    entity_distribution = pd.Series([
        ent['label'] for result in results for ent in result['entities']
    ]).value_counts()
    
    relation_distribution = pd.Series([
        rel['relation'] for result in results for rel in result['relations']
    ]).value_counts()
    
    event_distribution = pd.Series([
        event['type'] for result in results for event in result['events']
    ]).value_counts()
    
    # Plot distributions
    plot_entity_distribution([ent for result in results for ent in result['entities']])
    
    # Create summary
    summary = {
        'entity_distribution': entity_distribution.to_dict(),
        'relation_distribution': relation_distribution.to_dict(),
        'event_distribution': event_distribution.to_dict()
    }
    
    return summary
```

## Next Steps
1. Learn about advanced NER techniques
2. Study relation extraction methods
3. Explore event extraction
4. Practice with real-world applications
5. Learn about knowledge graph construction

## Resources
- [spaCy NER](https://spacy.io/usage/linguistic-features#named-entities)
- [Stanford OpenIE](https://nlp.stanford.edu/software/openie.html)
- [AllenNLP](https://allennlp.org/)
- [Hugging Face Transformers](https://huggingface.co/transformers/) 