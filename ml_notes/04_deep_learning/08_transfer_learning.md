# Transfer Learning

## Background and Introduction
Transfer Learning is a machine learning technique where knowledge gained from training on one task is applied to a different but related task. It leverages pre-trained models to improve learning efficiency and performance on new tasks, especially when limited training data is available.

## What is Transfer Learning?
Transfer Learning is characterized by:
1. Knowledge transfer from source to target domain
2. Feature reuse and adaptation
3. Domain adaptation
4. Fine-tuning capabilities
5. Model generalization

## Why Transfer Learning?
1. **Reduced Data Requirements**: Work with smaller datasets
2. **Improved Performance**: Better results with less training
3. **Faster Training**: Leverage pre-trained models
4. **Domain Adaptation**: Apply knowledge across domains
5. **Resource Efficiency**: Save computational resources

## How to Implement Transfer Learning?

### 1. Basic Transfer Learning
```python
import tensorflow as tf
from tensorflow.keras import layers, models, applications
import numpy as np
import matplotlib.pyplot as plt

def create_transfer_model(base_model_name='VGG16', num_classes=10):
    # Load pre-trained model
    if base_model_name == 'VGG16':
        base_model = applications.VGG16(weights='imagenet', include_top=False)
    elif base_model_name == 'ResNet50':
        base_model = applications.ResNet50(weights='imagenet', include_top=False)
    else:
        raise ValueError(f"Unsupported base model: {base_model_name}")
    
    # Freeze base model layers
    base_model.trainable = False
    
    # Create new model
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model

def fine_tune_model(model, num_layers_to_unfreeze=3):
    # Unfreeze the last few layers
    for layer in model.layers[0].layers[-num_layers_to_unfreeze:]:
        layer.trainable = True
    
    # Recompile model with lower learning rate
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-5),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model
```

### 2. Domain Adaptation
```python
class DomainAdaptationModel:
    def __init__(self, input_shape, num_classes):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.feature_extractor = self._build_feature_extractor()
        self.classifier = self._build_classifier()
        self.domain_classifier = self._build_domain_classifier()
    
    def _build_feature_extractor(self):
        model = models.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=self.input_shape),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu')
        ])
        return model
    
    def _build_classifier(self):
        model = models.Sequential([
            layers.Flatten(),
            layers.Dense(64, activation='relu'),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        return model
    
    def _build_domain_classifier(self):
        model = models.Sequential([
            layers.Flatten(),
            layers.Dense(64, activation='relu'),
            layers.Dense(1, activation='sigmoid')
        ])
        return model
    
    def build_model(self):
        inputs = layers.Input(shape=self.input_shape)
        features = self.feature_extractor(inputs)
        class_output = self.classifier(features)
        domain_output = self.domain_classifier(features)
        
        model = models.Model(inputs=inputs,
                           outputs=[class_output, domain_output])
        return model

def train_domain_adaptation(model, source_data, target_data, epochs=10):
    # Prepare data
    source_images, source_labels = source_data
    target_images, _ = target_data
    
    # Create domain labels
    source_domain = np.ones((len(source_images), 1))
    target_domain = np.zeros((len(target_images), 1))
    
    # Training loop
    for epoch in range(epochs):
        # Train on source domain
        source_loss = model.train_on_batch(
            source_images,
            [source_labels, source_domain]
        )
        
        # Train on target domain
        target_loss = model.train_on_batch(
            target_images,
            [np.zeros((len(target_images), model.num_classes)), target_domain]
        )
        
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"Source Loss: {source_loss}")
        print(f"Target Loss: {target_loss}")
```

### 3. Feature Extraction and Fine-tuning
```python
def extract_features(model, data, layer_name):
    # Create feature extractor
    feature_extractor = models.Model(
        inputs=model.input,
        outputs=model.get_layer(layer_name).output
    )
    
    # Extract features
    features = feature_extractor.predict(data)
    return features

def fine_tune_model(model, train_data, val_data, epochs=10):
    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-5),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Train model
    history = model.fit(
        train_data,
        validation_data=val_data,
        epochs=epochs,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=3,
                restore_best_weights=True
            )
        ]
    )
    
    return history
```

## Model Evaluation

### 1. Performance Metrics
```python
def evaluate_transfer_learning(model, test_data):
    # Evaluate model
    test_loss, test_accuracy = model.evaluate(test_data)
    
    # Get predictions
    predictions = model.predict(test_data)
    y_pred = np.argmax(predictions, axis=1)
    y_true = test_data.classes
    
    # Calculate metrics
    from sklearn.metrics import classification_report, confusion_matrix
    report = classification_report(y_true, y_pred)
    conf_matrix = confusion_matrix(y_true, y_pred)
    
    return {
        'test_loss': test_loss,
        'test_accuracy': test_accuracy,
        'classification_report': report,
        'confusion_matrix': conf_matrix
    }

def plot_training_history(history):
    plt.figure(figsize=(12, 4))
    
    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'])
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'])
    
    plt.tight_layout()
    plt.show()
```

## Common Interview Questions

1. **Q: What is the difference between transfer learning and fine-tuning?**
   - A: Transfer learning involves using a pre-trained model as a starting point for a new task, while fine-tuning specifically refers to the process of adjusting the pre-trained model's weights for the new task. Fine-tuning is a step in transfer learning that can be done after initial transfer.

2. **Q: When should you use transfer learning?**
   - A: Transfer learning is beneficial when:
     - Limited training data is available
     - The new task is similar to the pre-trained model's task
     - Computational resources are limited
     - Quick deployment is required
     - The target domain has some similarity to the source domain

3. **Q: How do you choose which layers to fine-tune?**
   - A: Considerations include:
     - Task similarity to pre-trained model
     - Amount of available training data
     - Computational resources
     - Desired level of adaptation
     - Model architecture and layer roles

## Hands-on Task: Image Classification

### Project: Flower Classification
```python
def flower_classification_project():
    # Load and preprocess data
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        'flowers',
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(224, 224),
        batch_size=32
    )
    
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        'flowers',
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(224, 224),
        batch_size=32
    )
    
    # Create and compile model
    model = create_transfer_model('VGG16', num_classes=5)
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Train model
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=10
    )
    
    # Fine-tune model
    model = fine_tune_model(model, num_layers_to_unfreeze=3)
    history_fine = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=5
    )
    
    # Evaluate model
    results = evaluate_transfer_learning(model, val_ds)
    
    # Plot training history
    plot_training_history(history)
    plot_training_history(history_fine)
    
    return {
        'model': model,
        'history': history,
        'history_fine': history_fine,
        'evaluation': results
    }
```

## Next Steps
1. Learn about advanced transfer learning techniques
2. Study domain adaptation methods
3. Explore multi-task learning
4. Practice with real-world applications
5. Learn about model optimization

## Resources
- [TensorFlow Transfer Learning Guide](https://www.tensorflow.org/tutorials/images/transfer_learning)
- [Transfer Learning Survey](https://arxiv.org/abs/1911.02685)
- [Domain Adaptation Tutorial](https://www.coursera.org/learn/domain-adaptation)
- [Transfer Learning in Deep Learning](https://www.deeplearning.ai/) 