# Computer Vision

## Background and Introduction
Computer Vision is a field of artificial intelligence that enables computers to interpret and understand visual information from the world. It combines image processing, machine learning, and deep learning to analyze and extract meaningful information from images and videos. Computer Vision has become increasingly important with applications in autonomous vehicles, medical imaging, security systems, and augmented reality.

## What is Computer Vision?
Computer Vision involves several key tasks:
1. Image Classification
2. Object Detection
3. Image Segmentation
4. Face Recognition
5. Image Generation
6. Video Analysis
7. 3D Reconstruction

## Why Computer Vision?
1. **Automation**: Automate visual inspection tasks
2. **Understanding**: Enable machines to understand visual data
3. **Analysis**: Extract insights from images and videos
4. **Interaction**: Enable human-computer interaction through visual means
5. **Innovation**: Drive new applications and technologies

## How to Process Images?

### 1. Image Preprocessing
```python
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import exposure
from skimage.filters import threshold_otsu

def preprocess_image(image_path, size=(224, 224)):
    # Read image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Resize image
    image = cv2.resize(image, size)
    
    # Normalize pixel values
    image = image.astype(np.float32) / 255.0
    
    return image

def enhance_image(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Apply histogram equalization
    equalized = exposure.equalize_hist(gray)
    
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    
    # Apply edge detection
    edges = cv2.Canny(gray, 100, 200)
    
    return {
        'original': image,
        'gray': gray,
        'equalized': equalized,
        'blurred': blurred,
        'edges': edges
    }

def segment_image(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Apply Otsu's thresholding
    thresh = threshold_otsu(gray)
    binary = gray > thresh
    
    # Find contours
    contours, _ = cv2.findContours(
        binary.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )
    
    # Draw contours
    segmented = image.copy()
    cv2.drawContours(segmented, contours, -1, (0, 255, 0), 2)
    
    return segmented

# Example usage
def demonstrate_preprocessing():
    # Load sample image
    image = cv2.imread('sample.jpg')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Enhance image
    enhanced = enhance_image(image)
    
    # Segment image
    segmented = segment_image(image)
    
    # Display results
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes[0, 0].imshow(enhanced['original'])
    axes[0, 0].set_title('Original')
    axes[0, 1].imshow(enhanced['gray'], cmap='gray')
    axes[0, 1].set_title('Grayscale')
    axes[0, 2].imshow(enhanced['equalized'], cmap='gray')
    axes[0, 2].set_title('Equalized')
    axes[1, 0].imshow(enhanced['blurred'])
    axes[1, 0].set_title('Blurred')
    axes[1, 1].imshow(enhanced['edges'], cmap='gray')
    axes[1, 1].set_title('Edges')
    axes[1, 2].imshow(segmented)
    axes[1, 2].set_title('Segmented')
    plt.tight_layout()
    plt.show()
    
    return enhanced, segmented
```

### 2. Feature Extraction
```python
from skimage.feature import hog, local_binary_pattern
from skimage import data, exposure

def extract_features(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Extract HOG features
    hog_features, hog_image = hog(
        gray,
        orientations=8,
        pixels_per_cell=(16, 16),
        cells_per_block=(1, 1),
        visualize=True
    )
    
    # Extract LBP features
    lbp = local_binary_pattern(gray, 8, 1, method='uniform')
    
    # Extract color histogram
    hist = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    
    return {
        'hog_features': hog_features,
        'hog_image': hog_image,
        'lbp': lbp,
        'color_hist': hist
    }

# Example usage
def demonstrate_feature_extraction():
    # Load sample image
    image = cv2.imread('sample.jpg')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Extract features
    features = extract_features(image)
    
    # Display results
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes[0, 0].imshow(image)
    axes[0, 0].set_title('Original')
    axes[0, 1].imshow(features['hog_image'], cmap='gray')
    axes[0, 1].set_title('HOG Features')
    axes[1, 0].imshow(features['lbp'], cmap='gray')
    axes[1, 0].set_title('LBP Features')
    axes[1, 1].plot(features['color_hist'])
    axes[1, 1].set_title('Color Histogram')
    plt.tight_layout()
    plt.show()
    
    return features
```

### 3. Image Classification
```python
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D

def create_classification_model(num_classes):
    # Load pre-trained ResNet50 model
    base_model = ResNet50(weights='imagenet', include_top=False)
    
    # Add custom classification layers
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    
    # Create model
    model = Model(inputs=base_model.input, outputs=predictions)
    
    # Compile model
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def train_classifier(model, train_dir, validation_dir, batch_size=32, epochs=10):
    # Data augmentation
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    validation_datagen = ImageDataGenerator(rescale=1./255)
    
    # Create data generators
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='categorical'
    )
    
    validation_generator = validation_datagen.flow_from_directory(
        validation_dir,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='categorical'
    )
    
    # Train model
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // batch_size,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // batch_size
    )
    
    return history

# Example usage
def demonstrate_classification():
    # Create model
    model = create_classification_model(num_classes=10)
    
    # Train model
    history = train_classifier(
        model,
        train_dir='train_data',
        validation_dir='validation_data'
    )
    
    return model, history
```

### 4. Object Detection
```python
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.models import Model

def create_detection_model(num_classes):
    # Load pre-trained ResNet50 model
    base_model = ResNet50(weights='imagenet', include_top=False)
    
    # Add detection layers
    x = base_model.output
    x = Conv2D(256, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Flatten()(x)
    
    # Classification head
    classification = Dense(num_classes, activation='softmax', name='classification')(x)
    
    # Regression head for bounding boxes
    regression = Dense(4, name='regression')(x)
    
    # Create model
    model = Model(inputs=base_model.input, outputs=[classification, regression])
    
    # Compile model
    model.compile(
        optimizer='adam',
        loss={
            'classification': 'categorical_crossentropy',
            'regression': 'mse'
        },
        loss_weights={
            'classification': 1.0,
            'regression': 1.0
        }
    )
    
    return model

def detect_objects(model, image, confidence_threshold=0.5):
    # Preprocess image
    processed_image = preprocess_image(image)
    
    # Make prediction
    classification, regression = model.predict(np.expand_dims(processed_image, axis=0))
    
    # Get class with highest confidence
    class_id = np.argmax(classification[0])
    confidence = classification[0][class_id]
    
    # Get bounding box
    bbox = regression[0]
    
    if confidence > confidence_threshold:
        # Draw bounding box
        x, y, w, h = bbox
        cv2.rectangle(
            image,
            (int(x), int(y)),
            (int(x + w), int(y + h)),
            (0, 255, 0),
            2
        )
        
        # Add label
        cv2.putText(
            image,
            f'Class {class_id}: {confidence:.2f}',
            (int(x), int(y - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2
        )
    
    return image

# Example usage
def demonstrate_detection():
    # Create model
    model = create_detection_model(num_classes=10)
    
    # Load image
    image = cv2.imread('sample.jpg')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Detect objects
    result = detect_objects(model, image)
    
    # Display result
    plt.figure(figsize=(10, 8))
    plt.imshow(result)
    plt.title('Object Detection')
    plt.show()
    
    return result
```

## Model Evaluation

### 1. Classification Metrics
```python
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

def evaluate_classifier(y_true, y_pred, class_names):
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()
    
    # Classification report
    print(classification_report(y_true, y_pred, target_names=class_names))
```

### 2. Detection Metrics
```python
def calculate_iou(box1, box2):
    # Calculate intersection
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    
    # Calculate union
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = box1_area + box2_area - intersection
    
    # Calculate IoU
    iou = intersection / union if union > 0 else 0
    
    return iou

def evaluate_detector(predictions, ground_truth, iou_threshold=0.5):
    # Calculate IoU for each prediction
    ious = []
    for pred_box, gt_box in zip(predictions, ground_truth):
        iou = calculate_iou(pred_box, gt_box)
        ious.append(iou)
    
    # Calculate precision and recall
    true_positives = sum(iou >= iou_threshold for iou in ious)
    precision = true_positives / len(predictions)
    recall = true_positives / len(ground_truth)
    
    # Calculate F1 score
    f1 = 2 * (precision * recall) / (precision + recall)
    
    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'ious': ious
    }
```

## Common Interview Questions

1. **Q: What is the difference between image classification and object detection?**
   - A: Image classification assigns a single label to an entire image, while object detection identifies and locates multiple objects within an image by drawing bounding boxes around them. Object detection is more complex as it requires both classification and localization.

2. **Q: What are the main challenges in computer vision?**
   - A: Key challenges include:
     - Variations in lighting and viewpoint
     - Occlusion and clutter
     - Scale and rotation invariance
     - Real-time processing requirements
     - Limited training data
     - Computational complexity

3. **Q: How do you handle overfitting in computer vision models?**
   - A: Several approaches can be used:
     - Data augmentation
     - Transfer learning
     - Regularization techniques
     - Dropout layers
     - Early stopping
     - Cross-validation

## Hands-on Task: Image Classification Project

### Project: Plant Disease Classification
```python
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
import matplotlib.pyplot as plt
import numpy as np

# Create and compile model
def create_plant_disease_model(num_classes):
    base_model = ResNet50(weights='imagenet', include_top=False)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# Data augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

validation_datagen = ImageDataGenerator(rescale=1./255)

# Create data generators
train_generator = train_datagen.flow_from_directory(
    'plant_disease/train',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

validation_generator = validation_datagen.flow_from_directory(
    'plant_disease/validation',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

# Create and train model
model = create_plant_disease_model(num_classes=len(train_generator.class_indices))
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // 32,
    epochs=10,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // 32
)

# Plot training history
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'])

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'])
plt.show()

# Evaluate model
test_generator = validation_datagen.flow_from_directory(
    'plant_disease/test',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

test_loss, test_acc = model.evaluate(test_generator)
print(f'Test accuracy: {test_acc:.4f}')
```

## Next Steps
1. Learn about advanced computer vision models (YOLO, Faster R-CNN)
2. Study image segmentation techniques
3. Explore video analysis
4. Practice with real-world datasets
5. Learn about 3D computer vision

## Resources
- [OpenCV Documentation](https://docs.opencv.org/)
- [TensorFlow Computer Vision Tutorials](https://www.tensorflow.org/tutorials/images)
- [PyTorch Computer Vision Tutorials](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html)
- [Computer Vision: Algorithms and Applications](https://www.amazon.com/Computer-Vision-Algorithms-Applications-Science/dp/1848829345) 