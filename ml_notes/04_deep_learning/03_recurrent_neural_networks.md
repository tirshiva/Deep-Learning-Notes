# Recurrent Neural Networks (RNNs)

## Background and Introduction
Recurrent Neural Networks (RNNs) are a class of neural networks designed to process sequential data by maintaining an internal memory of previous inputs. They are particularly effective for tasks involving time series, natural language processing, and speech recognition. RNNs can capture temporal dependencies and patterns in sequential data, making them essential for many real-world applications.

## What are RNNs?
RNNs are characterized by:
1. Sequential data processing
2. Internal memory (hidden state)
3. Parameter sharing across time steps
4. Ability to handle variable-length sequences
5. Temporal dependency modeling

## Why RNNs?
1. **Sequence Modeling**: Handle ordered data effectively
2. **Memory**: Maintain information about previous inputs
3. **Variable Length**: Process sequences of different lengths
4. **Temporal Patterns**: Capture time-dependent relationships
5. **Wide Applications**: Used in NLP, time series, and more

## How to Implement RNNs?

### 1. Basic RNN Architecture
```python
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

def create_basic_rnn(input_shape, num_classes):
    model = models.Sequential([
        # RNN layers
        layers.SimpleRNN(64, return_sequences=True, input_shape=input_shape),
        layers.Dropout(0.2),
        layers.SimpleRNN(32),
        layers.Dropout(0.2),
        
        # Dense layers
        layers.Dense(32, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model

def compile_model(model):
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model
```

### 2. LSTM Architecture
```python
def create_lstm_model(input_shape, num_classes):
    model = models.Sequential([
        # LSTM layers
        layers.LSTM(64, return_sequences=True, input_shape=input_shape),
        layers.Dropout(0.2),
        layers.LSTM(32),
        layers.Dropout(0.2),
        
        # Dense layers
        layers.Dense(32, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model

def create_bidirectional_lstm(input_shape, num_classes):
    model = models.Sequential([
        # Bidirectional LSTM layers
        layers.Bidirectional(
            layers.LSTM(64, return_sequences=True),
            input_shape=input_shape
        ),
        layers.Dropout(0.2),
        layers.Bidirectional(layers.LSTM(32)),
        layers.Dropout(0.2),
        
        # Dense layers
        layers.Dense(32, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model
```

### 3. GRU Architecture
```python
def create_gru_model(input_shape, num_classes):
    model = models.Sequential([
        # GRU layers
        layers.GRU(64, return_sequences=True, input_shape=input_shape),
        layers.Dropout(0.2),
        layers.GRU(32),
        layers.Dropout(0.2),
        
        # Dense layers
        layers.Dense(32, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model
```

## Model Training and Evaluation

### 1. Training Process
```python
def train_rnn_model(model, X_train, y_train, X_val, y_val, epochs=10):
    # Create callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=3,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=2
        )
    ]
    
    # Train model
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=32,
        validation_data=(X_val, y_val),
        callbacks=callbacks
    )
    
    return history

def plot_training_history(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot accuracy
    ax1.plot(history.history['accuracy'], label='Training')
    ax1.plot(history.history['val_accuracy'], label='Validation')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    
    # Plot loss
    ax2.plot(history.history['loss'], label='Training')
    ax2.plot(history.history['val_loss'], label='Validation')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    
    plt.tight_layout()
    plt.show()
```

### 2. Model Evaluation
```python
def evaluate_model(model, X_test, y_test):
    # Get predictions
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    # Calculate metrics
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_classes))
    
    # Plot confusion matrix
    cm = confusion_matrix(y_test, y_pred_classes)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()
    
    return {
        'predictions': y_pred,
        'predicted_classes': y_pred_classes,
        'confusion_matrix': cm
    }
```

### 3. Sequence Generation
```python
def generate_sequence(model, seed_sequence, length):
    sequence = seed_sequence.copy()
    
    for _ in range(length):
        # Get prediction for next step
        next_step = model.predict(sequence[-1:])
        
        # Add to sequence
        sequence = np.append(sequence, next_step, axis=0)
    
    return sequence

def plot_sequence(sequence, title="Generated Sequence"):
    plt.figure(figsize=(10, 4))
    plt.plot(sequence)
    plt.title(title)
    plt.xlabel('Time Step')
    plt.ylabel('Value')
    plt.show()
```

## Common Interview Questions

1. **Q: What is the difference between RNN, LSTM, and GRU?**
   - A: RNNs are the basic architecture with a simple recurrent unit. LSTMs add memory cells and gates to handle long-term dependencies. GRUs are a simplified version of LSTMs with fewer parameters but similar performance in many cases.

2. **Q: What is the vanishing gradient problem in RNNs?**
   - A: The vanishing gradient problem occurs when gradients become very small during backpropagation through time, making it difficult for the network to learn long-term dependencies. LSTMs and GRUs were designed to address this issue.

3. **Q: When should you use bidirectional RNNs?**
   - A: Bidirectional RNNs are useful when:
     - Context from both past and future is important
     - The task requires understanding the full sequence
     - You have enough computational resources
     - The sequence length is not too long

## Hands-on Task: Time Series Prediction

### Project: Stock Price Prediction
```python
def stock_price_prediction_project():
    # Generate sample stock data
    def generate_stock_data(n_steps=1000):
        t = np.linspace(0, 100, n_steps)
        price = np.sin(0.1 * t) + np.random.normal(0, 0.1, n_steps)
        return price
    
    # Prepare sequences
    def prepare_sequences(data, seq_length):
        X, y = [], []
        for i in range(len(data) - seq_length):
            X.append(data[i:i + seq_length])
            y.append(data[i + seq_length])
        return np.array(X), np.array(y)
    
    # Generate data
    data = generate_stock_data()
    seq_length = 20
    X, y = prepare_sequences(data, seq_length)
    
    # Split data
    train_size = int(0.8 * len(X))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # Reshape data for RNN
    X_train = X_train.reshape(-1, seq_length, 1)
    X_test = X_test.reshape(-1, seq_length, 1)
    
    # Create model
    model = create_lstm_model((seq_length, 1), 1)
    model = compile_model(model)
    
    # Train model
    history = train_rnn_model(model, X_train, y_train, X_test, y_test)
    
    # Plot training history
    plot_training_history(history)
    
    # Generate predictions
    predictions = model.predict(X_test)
    
    # Plot results
    plt.figure(figsize=(12, 6))
    plt.plot(y_test, label='Actual')
    plt.plot(predictions, label='Predicted')
    plt.title('Stock Price Prediction')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.show()
    
    return {
        'model': model,
        'history': history,
        'predictions': predictions,
        'actual': y_test
    }
```

## Next Steps
1. Learn about attention mechanisms
2. Study transformer architectures
3. Explore sequence-to-sequence models
4. Practice with real-world datasets
5. Learn about model optimization

## Resources
- [TensorFlow RNN Guide](https://www.tensorflow.org/tutorials/text/text_classification_rnn)
- [Understanding LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)
- [Deep Learning Book](https://www.deeplearningbook.org/)
- [Keras Documentation](https://keras.io/) 