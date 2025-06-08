# Autoencoders

## Background and Introduction
Autoencoders are neural networks designed to learn efficient representations of data by training the network to reconstruct the input from a compressed representation. They consist of an encoder that compresses the input into a latent space and a decoder that reconstructs the input from the latent representation. Autoencoders are widely used for dimensionality reduction, feature learning, and anomaly detection.

## What are Autoencoders?
Autoencoders are characterized by:
1. Encoder network for compression
2. Decoder network for reconstruction
3. Latent space representation
4. Reconstruction loss
5. Unsupervised learning approach

## Why Autoencoders?
1. **Dimensionality Reduction**: Compress data efficiently
2. **Feature Learning**: Discover meaningful representations
3. **Anomaly Detection**: Identify unusual patterns
4. **Denoising**: Remove noise from data
5. **Data Generation**: Create new samples

## How to Implement Autoencoders?

### 1. Basic Autoencoder Architecture
```python
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

def create_basic_autoencoder(input_shape, latent_dim):
    # Encoder
    encoder = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(latent_dim, activation='relu')
    ])
    
    # Decoder
    decoder = models.Sequential([
        layers.Input(shape=(latent_dim,)),
        layers.Dense(64, activation='relu'),
        layers.Dense(128, activation='relu'),
        layers.Dense(np.prod(input_shape), activation='sigmoid'),
        layers.Reshape(input_shape)
    ])
    
    # Autoencoder
    autoencoder = models.Sequential([encoder, decoder])
    
    return encoder, decoder, autoencoder

def compile_autoencoder(autoencoder):
    autoencoder.compile(
        optimizer='adam',
        loss='binary_crossentropy'
    )
    return autoencoder
```

### 2. Convolutional Autoencoder
```python
def create_convolutional_autoencoder(input_shape, latent_dim):
    # Encoder
    encoder = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same', strides=2),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same', strides=2),
        layers.Conv2D(128, (3, 3), activation='relu', padding='same', strides=2),
        layers.Flatten(),
        layers.Dense(latent_dim, activation='relu')
    ])
    
    # Decoder
    decoder = models.Sequential([
        layers.Input(shape=(latent_dim,)),
        layers.Dense(4 * 4 * 128, activation='relu'),
        layers.Reshape((4, 4, 128)),
        layers.Conv2DTranspose(64, (3, 3), activation='relu', padding='same', strides=2),
        layers.Conv2DTranspose(32, (3, 3), activation='relu', padding='same', strides=2),
        layers.Conv2DTranspose(3, (3, 3), activation='sigmoid', padding='same', strides=2)
    ])
    
    # Autoencoder
    autoencoder = models.Sequential([encoder, decoder])
    
    return encoder, decoder, autoencoder
```

### 3. Variational Autoencoder
```python
class Sampling(layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

def create_variational_autoencoder(input_shape, latent_dim):
    # Encoder
    encoder_inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(32, 3, activation='relu', strides=2, padding='same')(encoder_inputs)
    x = layers.Conv2D(64, 3, activation='relu', strides=2, padding='same')(x)
    x = layers.Conv2D(128, 3, activation='relu', strides=2, padding='same')(x)
    x = layers.Flatten()(x)
    x = layers.Dense(256, activation='relu')(x)
    
    z_mean = layers.Dense(latent_dim, name='z_mean')(x)
    z_log_var = layers.Dense(latent_dim, name='z_log_var')(x)
    z = Sampling()([z_mean, z_log_var])
    
    encoder = models.Model(encoder_inputs, [z_mean, z_log_var, z], name='encoder')
    
    # Decoder
    latent_inputs = layers.Input(shape=(latent_dim,))
    x = layers.Dense(4 * 4 * 128, activation='relu')(latent_inputs)
    x = layers.Reshape((4, 4, 128))(x)
    x = layers.Conv2DTranspose(64, 3, activation='relu', strides=2, padding='same')(x)
    x = layers.Conv2DTranspose(32, 3, activation='relu', strides=2, padding='same')(x)
    decoder_outputs = layers.Conv2DTranspose(3, 3, activation='sigmoid', padding='same', strides=2)(x)
    
    decoder = models.Model(latent_inputs, decoder_outputs, name='decoder')
    
    # VAE
    outputs = decoder(encoder(encoder_inputs)[2])
    vae = models.Model(encoder_inputs, outputs, name='vae')
    
    # Add KL divergence loss
    kl_loss = -0.5 * tf.reduce_mean(
        z_log_var - tf.square(z_mean) - tf.exp(z_log_var) + 1)
    vae.add_loss(kl_loss)
    
    return encoder, decoder, vae
```

## Model Training and Evaluation

### 1. Training Process
```python
def train_autoencoder(autoencoder, train_data, val_data, epochs=50):
    # Create callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=3
        )
    ]
    
    # Train model
    history = autoencoder.fit(
        train_data,
        train_data,  # Autoencoders try to reconstruct their input
        epochs=epochs,
        batch_size=32,
        validation_data=(val_data, val_data),
        callbacks=callbacks
    )
    
    return history

def plot_training_history(history):
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Autoencoder Training History')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
```

### 2. Model Evaluation
```python
def evaluate_autoencoder(autoencoder, test_data):
    # Get reconstructions
    reconstructions = autoencoder.predict(test_data)
    
    # Calculate reconstruction error
    mse = tf.keras.losses.mean_squared_error(test_data, reconstructions)
    mse = tf.reduce_mean(mse, axis=[1, 2, 3])
    
    return {
        'reconstructions': reconstructions,
        'mse': mse
    }

def plot_reconstructions(original, reconstructed, n_samples=10):
    plt.figure(figsize=(20, 4))
    for i in range(n_samples):
        # Original
        plt.subplot(2, n_samples, i + 1)
        plt.imshow(original[i])
        plt.axis('off')
        
        # Reconstructed
        plt.subplot(2, n_samples, i + 1 + n_samples)
        plt.imshow(reconstructed[i])
        plt.axis('off')
    
    plt.show()
```

### 3. Latent Space Visualization
```python
def visualize_latent_space(encoder, data, labels):
    # Get latent representations
    latent_representations = encoder.predict(data)
    
    # If using VAE, take the mean
    if isinstance(latent_representations, list):
        latent_representations = latent_representations[0]
    
    # Reduce dimensionality for visualization
    from sklearn.manifold import TSNE
    tsne = TSNE(n_components=2)
    latent_2d = tsne.fit_transform(latent_representations)
    
    # Plot
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(latent_2d[:, 0], latent_2d[:, 1], c=labels, cmap='tab10')
    plt.colorbar(scatter)
    plt.title('Latent Space Visualization')
    plt.show()
```

## Common Interview Questions

1. **Q: What is the difference between autoencoders and PCA?**
   - A: While both perform dimensionality reduction:
     - Autoencoders can learn non-linear relationships
     - PCA is limited to linear transformations
     - Autoencoders can handle complex data structures
     - PCA is more interpretable
     - Autoencoders can be used for generation

2. **Q: How do variational autoencoders differ from regular autoencoders?**
   - A: Key differences include:
     - VAE learns a probability distribution in latent space
     - Regular autoencoders learn a deterministic mapping
     - VAE can generate new samples
     - VAE includes KL divergence loss
     - VAE has better regularization

3. **Q: What are the applications of autoencoders?**
   - A: Common applications include:
     - Dimensionality reduction
     - Feature learning
     - Anomaly detection
     - Image denoising
     - Data compression
     - Data generation

## Hands-on Task: Image Denoising

### Project: Denoising Autoencoder
```python
def denoising_autoencoder_project():
    # Load and preprocess data
    def load_and_preprocess_data():
        # Load MNIST dataset
        (x_train, _), (x_test, _) = tf.keras.datasets.mnist.load_data()
        
        # Normalize data
        x_train = x_train.astype('float32') / 255
        x_test = x_test.astype('float32') / 255
        
        # Add noise
        noise_factor = 0.5
        x_train_noisy = x_train + noise_factor * np.random.normal(
            loc=0.0, scale=1.0, size=x_train.shape)
        x_test_noisy = x_test + noise_factor * np.random.normal(
            loc=0.0, scale=1.0, size=x_test.shape)
        
        # Clip values
        x_train_noisy = np.clip(x_train_noisy, 0., 1.)
        x_test_noisy = np.clip(x_test_noisy, 0., 1.)
        
        return x_train_noisy, x_test_noisy, x_train, x_test
    
    # Create model
    def create_denoising_autoencoder():
        input_shape = (28, 28, 1)
        latent_dim = 32
        
        # Encoder
        encoder = models.Sequential([
            layers.Input(shape=input_shape),
            layers.Conv2D(32, (3, 3), activation='relu', padding='same', strides=2),
            layers.Conv2D(64, (3, 3), activation='relu', padding='same', strides=2),
            layers.Flatten(),
            layers.Dense(latent_dim, activation='relu')
        ])
        
        # Decoder
        decoder = models.Sequential([
            layers.Input(shape=(latent_dim,)),
            layers.Dense(7 * 7 * 64, activation='relu'),
            layers.Reshape((7, 7, 64)),
            layers.Conv2DTranspose(32, (3, 3), activation='relu', padding='same', strides=2),
            layers.Conv2DTranspose(1, (3, 3), activation='sigmoid', padding='same', strides=2)
        ])
        
        # Autoencoder
        autoencoder = models.Sequential([encoder, decoder])
        
        return encoder, decoder, autoencoder
    
    # Main execution
    x_train_noisy, x_test_noisy, x_train, x_test = load_and_preprocess_data()
    
    # Create model
    encoder, decoder, autoencoder = create_denoising_autoencoder()
    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
    
    # Train model
    history = train_autoencoder(autoencoder, x_train_noisy, x_test_noisy)
    
    # Evaluate model
    results = evaluate_autoencoder(autoencoder, x_test_noisy)
    
    # Plot results
    plot_training_history(history)
    plot_reconstructions(x_test_noisy, results['reconstructions'])
    
    return {
        'model': autoencoder,
        'history': history,
        'results': results
    }
```

## Next Steps
1. Learn about advanced autoencoder architectures
2. Study conditional autoencoders
3. Explore generative models
4. Practice with real-world datasets
5. Learn about model optimization

## Resources
- [Autoencoder Tutorial](https://www.tensorflow.org/tutorials/generative/autoencoder)
- [VAE Paper](https://arxiv.org/abs/1312.6114)
- [Deep Learning Book](https://www.deeplearningbook.org/)
- [Keras Documentation](https://keras.io/) 