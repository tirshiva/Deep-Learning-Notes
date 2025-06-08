# Question Answering and Chatbots

## Background and Introduction
Question Answering (QA) and Chatbots are advanced NLP applications that enable machines to understand and respond to human queries in natural language. These systems are crucial for information retrieval, customer service, and human-computer interaction.

## What are QA and Chatbots?
Key aspects include:
1. Question answering
2. Conversational AI
3. Natural language understanding
4. Response generation
5. Context management

## Why QA and Chatbots?
1. **Information Access**: Quick access to information
2. **Customer Service**: Automated customer support
3. **Efficiency**: 24/7 availability
4. **Scalability**: Handle multiple queries
5. **User Experience**: Natural interaction

## How to Implement QA and Chatbots?

### 1. Question Answering System
```python
import tensorflow as tf
from transformers import BertForQuestionAnswering, BertTokenizer
import numpy as np

class QuestionAnsweringSystem:
    def __init__(self, model_type='bert'):
        self.model_type = model_type
        if model_type == 'bert':
            self.model = self._load_bert_model()
        else:
            self.model = self._build_custom_model()
    
    def _load_bert_model(self):
        # Load pre-trained model
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertForQuestionAnswering.from_pretrained('bert-base-uncased')
        return self.model
    
    def _build_custom_model(self):
        # Create custom QA model
        model = tf.keras.Sequential([
            tf.keras.layers.Embedding(10000, 256),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(256, return_sequences=True)),
            tf.keras.layers.Attention(),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dense(2)  # Start and end positions
        ])
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def answer_question(self, question, context):
        if self.model_type == 'bert':
            # Tokenize input
            inputs = self.tokenizer(
                question,
                context,
                return_tensors='pt',
                max_length=512,
                truncation=True,
                padding=True
            )
            
            # Get model predictions
            outputs = self.model(**inputs)
            
            # Get start and end positions
            start_scores = outputs.start_logits
            end_scores = outputs.end_logits
            
            # Get answer span
            start_idx = torch.argmax(start_scores)
            end_idx = torch.argmax(end_scores)
            
            # Decode answer
            answer = self.tokenizer.decode(
                inputs['input_ids'][0][start_idx:end_idx+1]
            )
            return answer
        else:
            # Implement custom QA logic
            pass
```

### 2. Chatbot System
```python
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import tensorflow as tf
import numpy as np

class Chatbot:
    def __init__(self, model_type='gpt2'):
        self.model_type = model_type
        if model_type == 'gpt2':
            self.model = self._load_gpt2_model()
        else:
            self.model = self._build_custom_model()
        
        self.conversation_history = []
    
    def _load_gpt2_model(self):
        # Load pre-trained model
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.model = GPT2LMHeadModel.from_pretrained('gpt2')
        return self.model
    
    def _build_custom_model(self):
        # Create custom chatbot model
        model = tf.keras.Sequential([
            tf.keras.layers.Embedding(10000, 256),
            tf.keras.layers.LSTM(512, return_sequences=True),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.LSTM(256),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(10000, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def generate_response(self, user_input, max_length=100):
        # Add user input to conversation history
        self.conversation_history.append(f"User: {user_input}")
        
        if self.model_type == 'gpt2':
            # Prepare input
            conversation = " ".join(self.conversation_history)
            inputs = self.tokenizer.encode(
                conversation,
                return_tensors='pt',
                max_length=512,
                truncation=True
            )
            
            # Generate response
            outputs = self.model.generate(
                inputs,
                max_length=max_length,
                num_return_sequences=1,
                temperature=0.7,
                top_k=50,
                top_p=0.95,
                do_sample=True
            )
            
            # Decode response
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Add response to conversation history
            self.conversation_history.append(f"Bot: {response}")
            
            return response
        else:
            # Implement custom response generation
            pass
    
    def clear_history(self):
        self.conversation_history = []
```

### 3. Advanced QA System
```python
class AdvancedQASystem:
    def __init__(self):
        self.qa_model = self._build_qa_model()
        self.retriever = self._build_retriever()
    
    def _build_qa_model(self):
        # Create advanced QA model
        model = tf.keras.Sequential([
            tf.keras.layers.Embedding(10000, 256),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(512, return_sequences=True)),
            tf.keras.layers.Attention(),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dense(2)  # Start and end positions
        ])
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def _build_retriever(self):
        # Create document retriever
        from sklearn.feature_extraction.text import TfidfVectorizer
        return TfidfVectorizer(
            max_features=1000,
            stop_words='english'
        )
    
    def answer_with_context(self, question, documents):
        # Retrieve relevant documents
        relevant_docs = self._retrieve_relevant_docs(question, documents)
        
        # Combine relevant documents
        context = " ".join(relevant_docs)
        
        # Get answer
        answer = self.qa_model.answer_question(question, context)
        
        return {
            'answer': answer,
            'context': context,
            'relevant_docs': relevant_docs
        }
    
    def _retrieve_relevant_docs(self, question, documents):
        # Implement document retrieval logic
        pass
```

## Model Evaluation

### 1. Performance Metrics
```python
def evaluate_qa_system(model, test_data):
    # Calculate exact match
    exact_matches = 0
    total = len(test_data)
    
    for item in test_data:
        question = item['question']
        context = item['context']
        true_answer = item['answer']
        
        predicted_answer = model.answer_question(question, context)
        
        if predicted_answer.lower() == true_answer.lower():
            exact_matches += 1
    
    exact_match_score = exact_matches / total
    
    # Calculate F1 score
    f1_scores = []
    for item in test_data:
        question = item['question']
        context = item['context']
        true_answer = item['answer']
        
        predicted_answer = model.answer_question(question, context)
        
        f1 = calculate_f1_score(predicted_answer, true_answer)
        f1_scores.append(f1)
    
    avg_f1_score = sum(f1_scores) / len(f1_scores)
    
    return {
        'exact_match': exact_match_score,
        'f1_score': avg_f1_score
    }

def evaluate_chatbot(model, test_conversations):
    # Calculate response quality
    quality_scores = []
    
    for conversation in test_conversations:
        user_input = conversation['user_input']
        true_response = conversation['response']
        
        predicted_response = model.generate_response(user_input)
        
        # Calculate similarity score
        similarity = calculate_similarity(predicted_response, true_response)
        quality_scores.append(similarity)
    
    avg_quality_score = sum(quality_scores) / len(quality_scores)
    
    return {
        'response_quality': avg_quality_score
    }
```

## Common Interview Questions

1. **Q: What are the main approaches to question answering?**
   - A: Key approaches include:
     - Extractive QA
     - Generative QA
     - Hybrid approaches
     - Transformer-based models
     - Knowledge-based QA

2. **Q: How do you handle context in chatbots?**
   - A: Solutions include:
     - Conversation history
     - Context window
     - Memory mechanisms
     - State tracking
     - Entity recognition

3. **Q: What are the challenges in QA systems?**
   - A: Challenges include:
     - Context understanding
     - Answer accuracy
     - Handling ambiguity
     - Multi-hop reasoning
     - Factual consistency

## Hands-on Task: QA and Chatbot System

### Project: Intelligent Assistant
```python
def intelligent_assistant_project():
    # Initialize systems
    qa_system = QuestionAnsweringSystem(model_type='bert')
    chatbot = Chatbot(model_type='gpt2')
    advanced_qa = AdvancedQASystem()
    
    # Test QA system
    questions = [
        "What is machine learning?",
        "How does deep learning work?",
        "What are neural networks?"
    ]
    
    context = """
    Machine learning is a subset of artificial intelligence that focuses on developing systems that can learn from data.
    Deep learning is a type of machine learning that uses neural networks with multiple layers.
    Neural networks are computing systems inspired by the human brain's biological neural networks.
    """
    
    qa_results = []
    for question in questions:
        # Get answer
        answer = qa_system.answer_question(question, context)
        
        # Get advanced answer
        advanced_answer = advanced_qa.answer_with_context(question, [context])
        
        qa_results.append({
            'question': question,
            'answer': answer,
            'advanced_answer': advanced_answer
        })
    
    # Test chatbot
    conversations = [
        "Hello, how are you?",
        "What can you help me with?",
        "Tell me about AI"
    ]
    
    chat_results = []
    for user_input in conversations:
        # Generate response
        response = chatbot.generate_response(user_input)
        
        chat_results.append({
            'user_input': user_input,
            'response': response
        })
    
    # Evaluate results
    qa_metrics = evaluate_qa_system(qa_system, qa_results)
    chat_metrics = evaluate_chatbot(chatbot, chat_results)
    
    return {
        'qa_results': qa_results,
        'chat_results': chat_results,
        'qa_metrics': qa_metrics,
        'chat_metrics': chat_metrics
    }
```

## Next Steps
1. Learn about advanced QA techniques
2. Study chatbot architectures
3. Explore multi-turn conversations
4. Practice with real-world applications
5. Learn about model optimization

## Resources
- [Hugging Face Transformers](https://huggingface.co/transformers/)
- [BERT for QA](https://huggingface.co/bert-base-uncased)
- [GPT-2](https://huggingface.co/gpt2)
- [QA Guide](https://www.tensorflow.org/tutorials/text/transformer) 