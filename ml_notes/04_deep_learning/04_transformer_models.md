# Transformer Models

## Background and Introduction
Transformer models have revolutionized natural language processing and other sequence-based tasks by introducing a novel architecture based on self-attention mechanisms. They have become the foundation for state-of-the-art models like BERT, GPT, and T5, achieving superior performance in various tasks while being more parallelizable than traditional RNNs.

## What are Transformers?
Transformers are characterized by:
1. Self-attention mechanisms
2. Positional encoding
3. Multi-head attention
4. Feed-forward networks
5. Layer normalization

## Why Transformers?
1. **Parallelization**: Process entire sequences simultaneously
2. **Long-range Dependencies**: Better capture relationships
3. **Scalability**: Handle large datasets effectively
4. **Transfer Learning**: Pre-trained models for various tasks
5. **State-of-the-Art Performance**: Superior results in many domains

## How to Implement Transformers?

### 1. Basic Transformer Components
```python
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

def positional_encoding(position, d_model):
    angle_rads = np.arange(position)[:, np.newaxis] / np.power(10000, (2 * (np.arange(d_model)[np.newaxis, :] // 2)) / np.float32(d_model))
    
    # Apply sin to even indices
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    
    # Apply cos to odd indices
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    
    pos_encoding = angle_rads[np.newaxis, ...]
    return tf.cast(pos_encoding, dtype=tf.float32)

def scaled_dot_product_attention(q, k, v, mask):
    # Calculate attention weights
    matmul_qk = tf.matmul(q, k, transpose_b=True)
    
    # Scale matmul_qk
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
    
    # Add mask if provided
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)
    
    # Softmax is normalized on the last axis
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
    
    # Calculate output
    output = tf.matmul(attention_weights, v)
    
    return output, attention_weights

class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        
        assert d_model % self.num_heads == 0
        
        self.depth = d_model // self.num_heads
        
        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)
        
        self.dense = tf.keras.layers.Dense(d_model)
    
    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])
    
    def call(self, v, k, q, mask):
        batch_size = tf.shape(q)[0]
        
        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)
        
        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)
        
        scaled_attention, attention_weights = scaled_dot_product_attention(
            q, k, v, mask
        )
        
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model))
        
        output = self.dense(concat_attention)
        
        return output, attention_weights
```

### 2. Transformer Encoder
```python
def point_wise_feed_forward_network(d_model, dff):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(dff, activation='relu'),
        tf.keras.layers.Dense(d_model)
    ])

class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(EncoderLayer, self).__init__()
        
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = point_wise_feed_forward_network(d_model, dff)
        
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
    
    def call(self, x, training, mask):
        attn_output, _ = self.mha(x, x, x, mask)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)
        
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)
        
        return out2

class Encoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size,
                 maximum_position_encoding, rate=0.1):
        super(Encoder, self).__init__()
        
        self.d_model = d_model
        self.num_layers = num_layers
        
        self.embedding = tf.keras.layers.Embedding(input_vocab_size, d_model)
        self.pos_encoding = positional_encoding(maximum_position_encoding, d_model)
        
        self.enc_layers = [EncoderLayer(d_model, num_heads, dff, rate)
                          for _ in range(num_layers)]
        
        self.dropout = tf.keras.layers.Dropout(rate)
    
    def call(self, x, training, mask):
        seq_len = tf.shape(x)[1]
        
        # Add embedding and position encoding
        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]
        
        x = self.dropout(x, training=training)
        
        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training, mask)
        
        return x
```

### 3. Transformer Decoder
```python
class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(DecoderLayer, self).__init__()
        
        self.mha1 = MultiHeadAttention(d_model, num_heads)
        self.mha2 = MultiHeadAttention(d_model, num_heads)
        
        self.ffn = point_wise_feed_forward_network(d_model, dff)
        
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
        self.dropout3 = tf.keras.layers.Dropout(rate)
    
    def call(self, x, enc_output, training, look_ahead_mask, padding_mask):
        attn1, attn_weights_block1 = self.mha1(x, x, x, look_ahead_mask)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(attn1 + x)
        
        attn2, attn_weights_block2 = self.mha2(
            enc_output, enc_output, out1, padding_mask)
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(attn2 + out1)
        
        ffn_output = self.ffn(out2)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.layernorm3(ffn_output + out2)
        
        return out3, attn_weights_block1, attn_weights_block2

class Decoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, target_vocab_size,
                 maximum_position_encoding, rate=0.1):
        super(Decoder, self).__init__()
        
        self.d_model = d_model
        self.num_layers = num_layers
        
        self.embedding = tf.keras.layers.Embedding(target_vocab_size, d_model)
        self.pos_encoding = positional_encoding(maximum_position_encoding, d_model)
        
        self.dec_layers = [DecoderLayer(d_model, num_heads, dff, rate)
                          for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(rate)
    
    def call(self, x, enc_output, training, look_ahead_mask, padding_mask):
        seq_len = tf.shape(x)[1]
        attention_weights = {}
        
        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]
        
        x = self.dropout(x, training=training)
        
        for i in range(self.num_layers):
            x, block1, block2 = self.dec_layers[i](x, enc_output, training,
                                                 look_ahead_mask, padding_mask)
            
            attention_weights[f'decoder_layer{i+1}_block1'] = block1
            attention_weights[f'decoder_layer{i+1}_block2'] = block2
        
        return x, attention_weights
```

## Model Training and Evaluation

### 1. Training Process
```python
def create_masks(inp, tar):
    # Encoder padding mask
    enc_padding_mask = create_padding_mask(inp)
    
    # Decoder padding mask
    dec_padding_mask = create_padding_mask(inp)
    
    # Look ahead mask
    look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
    dec_target_padding_mask = create_padding_mask(tar)
    combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)
    
    return enc_padding_mask, combined_mask, dec_padding_mask

def create_padding_mask(seq):
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
    return seq[:, tf.newaxis, tf.newaxis, :]

def create_look_ahead_mask(size):
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask

def train_step(inp, tar, transformer, optimizer, loss_object, train_loss, train_accuracy):
    tar_inp = tar[:, :-1]
    tar_real = tar[:, 1:]
    
    enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp, tar_inp)
    
    with tf.GradientTape() as tape:
        predictions, _ = transformer(inp, tar_inp, True, enc_padding_mask,
                                  combined_mask, dec_padding_mask)
        loss = loss_function(tar_real, predictions, loss_object)
    
    gradients = tape.gradient(loss, transformer.trainable_variables)
    optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))
    
    train_loss(loss)
    train_accuracy(tar_real, predictions)
```

### 2. Model Evaluation
```python
def evaluate_model(transformer, test_dataset, max_length):
    results = []
    
    for (inp, tar) in test_dataset:
        result = translate(inp, transformer, max_length)
        results.append(result)
    
    return results

def translate(inp_sentence, transformer, max_length):
    # Tokenize input
    inp_sentence = tokenizer.texts_to_sequences([inp_sentence])[0]
    inp_sentence = tf.convert_to_tensor(inp_sentence)
    
    # Initialize output
    output = tf.convert_to_tensor([tokenizer.word_index['<start>']])
    
    for i in range(max_length):
        enc_padding_mask, combined_mask, dec_padding_mask = create_masks(
            inp_sentence, output)
        
        # Predictions
        predictions, attention_weights = transformer(
            inp_sentence, output, False, enc_padding_mask, combined_mask,
            dec_padding_mask)
        
        # Select last word
        predictions = predictions[:, -1:, :]
        predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)
        
        # Return if predicted end token
        if predicted_id == tokenizer.word_index['<end>']:
            return tf.squeeze(output, axis=0), attention_weights
        
        # Concatenate predicted word
        output = tf.concat([output, predicted_id], axis=-1)
    
    return tf.squeeze(output, axis=0), attention_weights
```

## Common Interview Questions

1. **Q: What is the key innovation of transformer models?**
   - A: The key innovation is the self-attention mechanism, which allows the model to weigh the importance of different words in a sequence when making predictions. This enables better capture of long-range dependencies and parallel processing of sequences.

2. **Q: How does multi-head attention work?**
   - A: Multi-head attention:
     - Splits the input into multiple heads
     - Computes attention for each head independently
     - Concatenates the results
     - Projects to the final output
     This allows the model to focus on different aspects of the input simultaneously.

3. **Q: What are the advantages of transformers over RNNs?**
   - A: Advantages include:
     - Parallel processing of sequences
     - Better handling of long-range dependencies
     - No vanishing gradient problems
     - More efficient training
     - Better performance on many tasks

## Hands-on Task: Text Translation

### Project: English to French Translation
```python
def translation_project():
    # Load and preprocess data
    def load_data():
        # Load English-French pairs
        pairs = load_eng_fra_dataset()
        
        # Create tokenizers
        eng_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')
        fra_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')
        
        # Fit tokenizers
        eng_tokenizer.fit_on_texts([pair[0] for pair in pairs])
        fra_tokenizer.fit_on_texts([pair[1] for pair in pairs])
        
        return pairs, eng_tokenizer, fra_tokenizer
    
    # Create transformer
    def create_transformer_model(vocab_size, d_model, num_heads, dff, num_layers, dropout_rate):
        transformer = Transformer(
            num_layers=num_layers,
            d_model=d_model,
            num_heads=num_heads,
            dff=dff,
            input_vocab_size=vocab_size,
            target_vocab_size=vocab_size,
            pe_input=vocab_size,
            pe_target=vocab_size,
            rate=dropout_rate
        )
        return transformer
    
    # Train model
    def train_transformer(transformer, train_dataset, epochs):
        # Initialize metrics
        train_loss = tf.keras.metrics.Mean(name='train_loss')
        train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
        
        # Initialize optimizer and loss
        optimizer = tf.keras.optimizers.Adam()
        loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
        
        # Training loop
        for epoch in range(epochs):
            train_loss.reset_states()
            train_accuracy.reset_states()
            
            for (inp, tar) in train_dataset:
                train_step(inp, tar, transformer, optimizer, loss_object,
                          train_loss, train_accuracy)
            
            print(f'Epoch {epoch + 1}, Loss: {train_loss.result():.4f}, '
                  f'Accuracy: {train_accuracy.result():.4f}')
    
    # Main execution
    pairs, eng_tokenizer, fra_tokenizer = load_data()
    
    # Create and train model
    transformer = create_transformer_model(
        vocab_size=10000,
        d_model=256,
        num_heads=8,
        dff=512,
        num_layers=4,
        dropout_rate=0.1
    )
    
    train_transformer(transformer, train_dataset, epochs=10)
    
    # Evaluate model
    test_results = evaluate_model(transformer, test_dataset, max_length=50)
    
    return {
        'model': transformer,
        'tokenizers': (eng_tokenizer, fra_tokenizer),
        'results': test_results
    }
```

## Next Steps
1. Learn about BERT and GPT models
2. Study transfer learning with transformers
3. Explore attention visualization
4. Practice with real-world datasets
5. Learn about model optimization

## Resources
- [Attention Is All You Need Paper](https://arxiv.org/abs/1706.03762)
- [TensorFlow Transformer Tutorial](https://www.tensorflow.org/tutorials/text/transformer)
- [Hugging Face Transformers](https://huggingface.co/transformers/)
- [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/) 