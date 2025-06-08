# Neural Networks and Deep Learning

## Background and Introduction
Neural Networks are computational models inspired by the human brain's structure and function. Deep Learning is a subset of machine learning that uses multiple layers of neural networks to learn hierarchical representations of data. The field has seen remarkable progress since the 2010s, leading to breakthroughs in various domains.

## What are Neural Networks?
Neural Networks are computational models composed of interconnected nodes (neurons) organized in layers. Each connection has an associated weight, and each neuron applies an activation function to its input. The network learns by adjusting these weights through a process called backpropagation.

## Why Neural Networks and Deep Learning?
1. **Powerful Learning**: Can learn complex patterns and relationships
2. **Feature Learning**: Automatically learns relevant features
3. **Versatility**: Applicable to various types of data
4. **State-of-the-art Performance**: Achieves best results in many domains
5. **Transfer Learning**: Can leverage pre-trained models

## How Do Neural Networks Work?

### 1. Basic Neural Network Implementation
```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class NeuralNetwork:
    def __init__(self, layer_sizes, learning_rate=0.01):
        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate
        self.weights = []
        self.biases = []
        
        # Initialize weights and biases
        for i in range(len(layer_sizes) - 1):
            self.weights.append(np.random.randn(layer_sizes[i], layer_sizes[i + 1]) * 0.01)
            self.biases.append(np.zeros((1, layer_sizes[i + 1])))
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, x):
        return x * (1 - x)
    
    def relu(self, x):
        return np.maximum(0, x)
    
    def relu_derivative(self, x):
        return np.where(x > 0, 1, 0)
    
    def forward(self, X):
        self.activations = [X]
        self.z_values = []
        
        for i in range(len(self.weights)):
            z = np.dot(self.activations[-1], self.weights[i]) + self.biases[i]
            self.z_values.append(z)
            
            if i == len(self.weights) - 1:
                # Output layer uses sigmoid
                activation = self.sigmoid(z)
            else:
                # Hidden layers use ReLU
                activation = self.relu(z)
            
            self.activations.append(activation)
        
        return self.activations[-1]
    
    def backward(self, X, y, output):
        m = X.shape[0]
        delta = output - y
        
        for i in range(len(self.weights) - 1, -1, -1):
            # Compute gradients
            dW = np.dot(self.activations[i].T, delta) / m
            db = np.sum(delta, axis=0, keepdims=True) / m
            
            if i > 0:
                # Compute delta for next layer
                delta = np.dot(delta, self.weights[i].T) * self.relu_derivative(self.activations[i])
            
            # Update weights and biases
            self.weights[i] -= self.learning_rate * dW
            self.biases[i] -= self.learning_rate * db
    
    def train(self, X, y, epochs):
        for epoch in range(epochs):
            # Forward pass
            output = self.forward(X)
            
            # Backward pass
            self.backward(X, y, output)
            
            # Print progress
            if epoch % 100 == 0:
                loss = np.mean(-y * np.log(output + 1e-8) - (1 - y) * np.log(1 - output + 1e-8))
                print(f"Epoch {epoch}, Loss: {loss:.4f}")
    
    def predict(self, X):
        return self.forward(X) > 0.5

# Generate sample data
X, y = make_moons(n_samples=1000, noise=0.1)
y = y.reshape(-1, 1)

# Split and scale data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create and train neural network
nn = NeuralNetwork(layer_sizes=[2, 4, 1], learning_rate=0.01)
nn.train(X_train_scaled, y_train, epochs=1000)

# Plot decision boundary
def plot_decision_boundary(X, y, model, title):
    h = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                        np.arange(y_min, y_max, h))
    
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    plt.figure(figsize=(10, 6))
    plt.contourf(xx, yy, Z, alpha=0.4)
    plt.scatter(X[:, 0], X[:, 1], c=y.ravel(), alpha=0.8)
    plt.title(title)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.show()

plot_decision_boundary(X_test_scaled, y_test, nn, 'Neural Network Decision Boundary')
```

### 2. Deep Learning with PyTorch
```python
import torch
import torch.nn as nn
import torch.optim as optim

class DeepNeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(DeepNeuralNetwork, self).__init__()
        
        # Create layers
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.Dropout(0.2))
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, output_size))
        layers.append(nn.Sigmoid())
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)

# Convert data to PyTorch tensors
X_train_tensor = torch.FloatTensor(X_train_scaled)
y_train_tensor = torch.FloatTensor(y_train)
X_test_tensor = torch.FloatTensor(X_test_scaled)
y_test_tensor = torch.FloatTensor(y_test)

# Create model
model = DeepNeuralNetwork(input_size=2, 
                         hidden_sizes=[16, 8, 4],
                         output_size=1)

# Define loss function and optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training loop
epochs = 1000
for epoch in range(epochs):
    # Forward pass
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    
    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # Print progress
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# Evaluate model
model.eval()
with torch.no_grad():
    y_pred = model(X_test_tensor)
    y_pred = (y_pred > 0.5).float()
    accuracy = (y_pred == y_test_tensor).float().mean()
    print(f"Test Accuracy: {accuracy:.4f}")
```

## Model Evaluation

### 1. Learning Curves
```python
def plot_learning_curves(train_losses, val_losses):
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Learning Curves')
    plt.legend()
    plt.grid(True)
    plt.show()
```

### 2. Model Architecture Visualization
```python
from torchviz import make_dot

def visualize_model_architecture(model, input_size):
    x = torch.randn(input_size)
    y = model(x)
    dot = make_dot(y, params=dict(model.named_parameters()))
    dot.render("model_architecture", format="png")
```

## Common Interview Questions

1. **Q: What is the difference between shallow and deep neural networks?**
   - A: Shallow networks have one hidden layer, while deep networks have multiple hidden layers. Deep networks can learn hierarchical features and are more powerful but require more data and computational resources.

2. **Q: What are common activation functions and when to use them?**
   - A: Common activation functions include:
     - ReLU: \(f(x) = max(0, x)\) - Most common, helps with vanishing gradient
     - Sigmoid: \(f(x) = \frac{1}{1 + e^{-x}}\) - For binary classification output
     - Tanh: \(f(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}\) - Zero-centered, for hidden layers
     - Softmax: \(f(x_i) = \frac{e^{x_i}}{\sum_j e^{x_j}}\) - For multi-class output

3. **Q: What are the advantages and disadvantages of deep learning?**
   - A: Advantages:
     - State-of-the-art performance
     - Automatic feature learning
     - Scalable to large datasets
     - Transfer learning capability
     Disadvantages:
     - Requires large amounts of data
     - Computationally expensive
     - Black box nature
     - Sensitive to hyperparameters

## Hands-on Task: Image Classification

### Project: MNIST Digit Recognition
```python
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Load MNIST dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST('data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('data', train=False, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000)

# Create CNN model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(1600, 100)
        self.fc2 = nn.Linear(100, 10)
    
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        x = self.conv2_drop(x)
        x = x.view(-1, 1600)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return torch.log_softmax(x, dim=1)

# Create model and move to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN().to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

# Training loop
def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

# Test function
def test():
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    
    test_loss /= len(test_loader.dataset)
    print(f'\nTest set: Average loss: {test_loss:.4f}, '
          f'Accuracy: {correct}/{len(test_loader.dataset)} '
          f'({100. * correct / len(test_loader.dataset):.0f}%)\n')

# Train model
epochs = 10
for epoch in range(1, epochs + 1):
    train(epoch)
    test()

# Visualize some predictions
def plot_predictions(model, test_loader, num_samples=5):
    model.eval()
    data, target = next(iter(test_loader))
    data, target = data.to(device), target.to(device)
    output = model(data)
    pred = output.argmax(dim=1, keepdim=True)
    
    plt.figure(figsize=(15, 3))
    for i in range(num_samples):
        plt.subplot(1, num_samples, i + 1)
        plt.imshow(data[i].cpu().squeeze(), cmap='gray')
        plt.title(f'True: {target[i]}\nPred: {pred[i].item()}')
        plt.axis('off')
    plt.show()

plot_predictions(model, test_loader)
```

## Next Steps
1. Learn about different neural network architectures
2. Study advanced deep learning techniques
3. Explore transfer learning
4. Practice with real-world datasets
5. Learn about model optimization

## Resources
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [Deep Learning Specialization](https://www.coursera.org/specializations/deep-learning)
- [Deep Learning with PyTorch](https://www.kaggle.com/learn/pytorch)
- [Deep Learning](https://www.amazon.com/Deep-Learning-Adaptive-Computation-Machine/dp/0262035618) 