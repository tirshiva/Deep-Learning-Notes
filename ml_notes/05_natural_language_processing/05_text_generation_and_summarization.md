# Text Generation and Summarization

## Background and Introduction
Text Generation and Summarization are advanced NLP tasks that involve creating new text or condensing existing text while preserving key information. These tasks are crucial for content creation, information extraction, and automated text processing.

## What are Text Generation and Summarization?
Key aspects include:
1. Text generation
2. Text summarization
3. Abstractive summarization
4. Extractive summarization
5. Content creation

## Why Text Generation and Summarization?
1. **Content Creation**: Generate new text content
2. **Information Condensation**: Create concise summaries
3. **Automation**: Automate content generation
4. **Efficiency**: Process large amounts of text
5. **Insights**: Extract key information

## How to Implement Text Generation and Summarization?

### 1. Text Generation
```python
import tensorflow as tf
from tensorflow.keras import layers, models
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import numpy as np

class TextGenerator:
    def __init__(self, model_type='transformer'):
        self.model_type = model_type
        if model_type == 'transformer':
            self.model = self._load_transformer_model()
        else:
            self.model = self._build_lstm_model()
    
    def _load_transformer_model(self):
        # Load pre-trained model
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.model = GPT2LMHeadModel.from_pretrained('gpt2')
        return self.model
    
    def _build_lstm_model(self):
        model = models.Sequential([
            layers.Embedding(10000, 256),
            layers.LSTM(512, return_sequences=True),
            layers.Dropout(0.2),
            layers.LSTM(256),
            layers.Dropout(0.2),
            layers.Dense(10000, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def generate_text(self, prompt, max_length=100, temperature=0.7):
        if self.model_type == 'transformer':
            # Tokenize input
            inputs = self.tokenizer(prompt, return_tensors='pt')
            
            # Generate text
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                num_return_sequences=1,
                temperature=temperature,
                top_k=50,
                top_p=0.95,
                do_sample=True
            )
            
            # Decode output
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return generated_text
        else:
            # Implement custom text generation logic
            pass
```

### 2. Text Summarization
```python
from transformers import T5ForConditionalGeneration, T5Tokenizer
import nltk
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

class TextSummarizer:
    def __init__(self, method='abstractive'):
        self.method = method
        if method == 'abstractive':
            self.model = self._load_abstractive_model()
        else:
            self.model = self._build_extractive_model()
    
    def _load_abstractive_model(self):
        # Load pre-trained model
        self.tokenizer = T5Tokenizer.from_pretrained('t5-base')
        self.model = T5ForConditionalGeneration.from_pretrained('t5-base')
        return self.model
    
    def _build_extractive_model(self):
        # Create TF-IDF vectorizer
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english'
        )
        return self.vectorizer
    
    def summarize(self, text, max_length=150, min_length=50):
        if self.method == 'abstractive':
            # Prepare input
            inputs = self.tokenizer.encode(
                "summarize: " + text,
                return_tensors='pt',
                max_length=512,
                truncation=True
            )
            
            # Generate summary
            outputs = self.model.generate(
                inputs,
                max_length=max_length,
                min_length=min_length,
                num_beams=4,
                length_penalty=2.0,
                early_stopping=True
            )
            
            # Decode output
            summary = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return summary
        else:
            # Implement extractive summarization
            sentences = sent_tokenize(text)
            tfidf_matrix = self.vectorizer.fit_transform(sentences)
            sentence_scores = np.sum(tfidf_matrix.toarray(), axis=1)
            
            # Select top sentences
            top_indices = np.argsort(sentence_scores)[-3:]
            summary = ' '.join([sentences[i] for i in sorted(top_indices)])
            return summary
```

### 3. Advanced Text Generation
```python
class AdvancedTextGenerator:
    def __init__(self):
        self.model = self._build_advanced_model()
    
    def _build_advanced_model(self):
        # Create advanced model with attention
        model = models.Sequential([
            layers.Embedding(10000, 256),
            layers.Bidirectional(layers.LSTM(512, return_sequences=True)),
            layers.Attention(),
            layers.Dropout(0.2),
            layers.LSTM(256),
            layers.Dropout(0.2),
            layers.Dense(10000, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def generate_with_style(self, prompt, style, max_length=100):
        # Add style conditioning
        style_embedding = self._get_style_embedding(style)
        
        # Generate text with style
        inputs = self._prepare_inputs(prompt, style_embedding)
        outputs = self.model.generate(inputs, max_length=max_length)
        
        return self._decode_outputs(outputs)
    
    def _get_style_embedding(self, style):
        # Get style embedding
        style_embeddings = {
            'formal': [1, 0, 0],
            'casual': [0, 1, 0],
            'creative': [0, 0, 1]
        }
        return style_embeddings.get(style, [0, 0, 0])
```

## Model Evaluation

### 1. Performance Metrics
```python
def evaluate_summarization(references, hypotheses):
    # Calculate ROUGE scores
    from rouge import Rouge
    rouge = Rouge()
    rouge_scores = rouge.get_scores(hypotheses, references, avg=True)
    
    # Calculate BLEU score
    from nltk.translate.bleu_score import corpus_bleu
    bleu_score = corpus_bleu([[ref] for ref in references], hypotheses)
    
    return {
        'rouge_scores': rouge_scores,
        'bleu_score': bleu_score
    }

def evaluate_text_generation(model, test_data):
    # Calculate perplexity
    perplexity = model.evaluate(test_data)
    
    # Generate sample text
    sample_text = model.generate_text("Once upon a time")
    
    return {
        'perplexity': perplexity,
        'sample_text': sample_text
    }
```

## Common Interview Questions

1. **Q: What are the main approaches to text summarization?**
   - A: Key approaches include:
     - Extractive summarization
     - Abstractive summarization
     - Hybrid approaches
     - Transformer-based models
     - Reinforcement learning

2. **Q: How do you handle long text in summarization?**
   - A: Solutions include:
     - Chunking
     - Hierarchical models
     - Attention mechanisms
     - Document structure analysis
     - Key information extraction

3. **Q: What are the challenges in text generation?**
   - A: Challenges include:
     - Coherence and consistency
     - Long-range dependencies
     - Style control
     - Factual accuracy
     - Bias and fairness

## Hands-on Task: Text Generation and Summarization

### Project: Article Summarization System
```python
def article_summarization_project():
    # Initialize models
    generator = TextGenerator(model_type='transformer')
    summarizer = TextSummarizer(method='abstractive')
    advanced_generator = AdvancedTextGenerator()
    
    # Test text generation
    prompts = [
        "Once upon a time",
        "The future of artificial intelligence",
        "Climate change and its impact"
    ]
    
    generated_texts = []
    for prompt in prompts:
        # Generate text
        text = generator.generate_text(prompt)
        
        # Generate text with style
        formal_text = advanced_generator.generate_with_style(prompt, 'formal')
        casual_text = advanced_generator.generate_with_style(prompt, 'casual')
        
        generated_texts.append({
            'prompt': prompt,
            'generated': text,
            'formal': formal_text,
            'casual': casual_text
        })
    
    # Test summarization
    articles = [
        "Long article about climate change...",
        "Detailed analysis of AI advancements...",
        "Comprehensive review of renewable energy..."
    ]
    
    summaries = []
    for article in articles:
        # Generate summary
        summary = summarizer.summarize(article)
        summaries.append({
            'original': article,
            'summary': summary
        })
    
    # Evaluate results
    references = [article for article in articles]
    hypotheses = [summary['summary'] for summary in summaries]
    metrics = evaluate_summarization(references, hypotheses)
    
    return {
        'generated_texts': generated_texts,
        'summaries': summaries,
        'metrics': metrics
    }
```

## Next Steps
1. Learn about advanced generation techniques
2. Study summarization methods
3. Explore style transfer
4. Practice with real-world applications
5. Learn about model optimization

## Resources
- [Hugging Face Transformers](https://huggingface.co/transformers/)
- [T5 Model](https://huggingface.co/t5-base)
- [GPT-2](https://huggingface.co/gpt2)
- [Text Generation Guide](https://www.tensorflow.org/tutorials/text/text_generation) 