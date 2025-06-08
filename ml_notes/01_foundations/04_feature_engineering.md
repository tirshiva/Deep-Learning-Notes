# Feature Engineering

## Background and Introduction
Feature engineering is the process of creating new features or modifying existing ones to improve model performance. It's often considered both an art and a science, as it requires domain knowledge, creativity, and technical expertise.

## What is Feature Engineering?
Feature engineering involves:
- Creating new features from existing data
- Transforming features to make them more useful
- Selecting the most relevant features
- Combining features to capture relationships
- Encoding features in appropriate formats

## Why Feature Engineering?
1. **Model Performance**: Better features lead to better model performance
2. **Data Understanding**: Helps understand relationships in data
3. **Computational Efficiency**: Reduces computational requirements
4. **Model Interpretability**: Makes models more interpretable
5. **Domain Knowledge**: Incorporates domain expertise into models

## How to Engineer Features?

### 1. Feature Creation

#### Time-Based Features
```python
import pandas as pd
import numpy as np

# Create sample time series data
df = pd.DataFrame({
    'timestamp': pd.date_range('2023-01-01', periods=100, freq='H'),
    'value': np.random.randn(100)
})

# Extract time components
df['hour'] = df['timestamp'].dt.hour
df['day'] = df['timestamp'].dt.day
df['month'] = df['timestamp'].dt.month
df['day_of_week'] = df['timestamp'].dt.dayofweek
df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)

# Create cyclical features
df['hour_sin'] = np.sin(2 * np.pi * df['hour']/24)
df['hour_cos'] = np.cos(2 * np.pi * df['hour']/24)
```

#### Interaction Features
```python
# Create sample data
df = pd.DataFrame({
    'age': [25, 30, 35, 40, 45],
    'income': [50000, 60000, 70000, 80000, 90000],
    'education_years': [12, 14, 16, 18, 20]
})

# Create interaction features
df['age_income'] = df['age'] * df['income']
df['income_per_year'] = df['income'] / df['education_years']
df['age_education'] = df['age'] * df['education_years']

# Create polynomial features
df['age_squared'] = df['age'] ** 2
df['income_squared'] = df['income'] ** 2
```

### 2. Feature Transformation

#### Log Transformation
```python
# Create sample data with skewed distribution
df = pd.DataFrame({
    'income': [1000, 2000, 5000, 10000, 50000, 100000]
})

# Apply log transformation
df['income_log'] = np.log1p(df['income'])  # log1p for handling zeros

# Apply box-cox transformation
from scipy import stats
df['income_boxcox'], lambda_param = stats.boxcox(df['income'])
```

#### Binning and Discretization
```python
# Create sample data
df = pd.DataFrame({
    'age': [20, 25, 30, 35, 40, 45, 50, 55, 60]
})

# Equal-width binning
df['age_bins'] = pd.cut(df['age'], bins=3, labels=['young', 'middle', 'old'])

# Equal-frequency binning
df['age_qbins'] = pd.qcut(df['age'], q=3, labels=['young', 'middle', 'old'])

# Custom binning
bins = [0, 30, 40, 50, 100]
labels = ['young', 'middle', 'senior', 'elderly']
df['age_custom'] = pd.cut(df['age'], bins=bins, labels=labels)
```

### 3. Feature Selection

#### Statistical Methods
```python
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# Create sample data
X = np.random.rand(100, 10)
y = np.random.randint(0, 2, 100)

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Select features using statistical tests
selector = SelectKBest(f_classif, k=5)
X_selected = selector.fit_transform(X_scaled, y)

# Get feature scores
scores = selector.scores_
```

#### Model-Based Methods
```python
# Using Random Forest
rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_scaled, y)

# Get feature importance
importances = rf.feature_importances_

# Select features based on importance
threshold = 0.1
selected_features = X_scaled[:, importances > threshold]
```

### 4. Advanced Feature Engineering

#### Text Features
```python
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Create sample text data
texts = [
    "Machine learning is fascinating",
    "Deep learning is a subset of machine learning",
    "Natural language processing is interesting"
]

# TF-IDF vectorization
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(texts)

# Text preprocessing
nltk.download('punkt')
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    tokens = [t for t in tokens if t not in stop_words]
    return ' '.join(tokens)

processed_texts = [preprocess_text(text) for text in texts]
```

#### Image Features
```python
from skimage import feature
import cv2

# Create sample image
image = np.random.rand(100, 100)

# Extract HOG features
hog_features = feature.hog(image, orientations=8, pixels_per_cell=(16, 16),
                          cells_per_block=(1, 1))

# Extract color histogram
hist = cv2.calcHist([image], [0], None, [256], [0, 256])
```

## Common Interview Questions

1. **Q: What are the key steps in feature engineering?**
   - A: Key steps include:
     - Understanding the data and domain
     - Creating new features
     - Transforming existing features
     - Selecting relevant features
     - Validating feature importance
     - Iterating and improving

2. **Q: How do you handle categorical variables with many categories?**
   - A: Common approaches include:
     - Target encoding
     - Frequency encoding
     - Hash encoding
     - Embedding layers (for deep learning)
     - Feature hashing

3. **Q: What are some common feature engineering techniques for time series data?**
   - A: Common techniques include:
     - Lag features
     - Rolling statistics
     - Seasonal decomposition
     - Time-based features
     - Fourier transforms

## Hands-on Task: Feature Engineering Project

### Project: House Price Prediction
```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.ensemble import RandomForestRegressor

# Create sample housing data
np.random.seed(42)
n_houses = 1000
data = {
    'area': np.random.normal(2000, 500, n_houses),
    'bedrooms': np.random.randint(1, 6, n_houses),
    'bathrooms': np.random.randint(1, 4, n_houses),
    'age': np.random.randint(0, 50, n_houses),
    'location': np.random.choice(['urban', 'suburban', 'rural'], n_houses),
    'price': np.random.normal(300000, 100000, n_houses)
}
df = pd.DataFrame(data)

# Feature engineering
# 1. Create interaction features
df['area_per_bedroom'] = df['area'] / df['bedrooms']
df['total_rooms'] = df['bedrooms'] + df['bathrooms']

# 2. Create polynomial features
df['area_squared'] = df['area'] ** 2
df['age_squared'] = df['age'] ** 2

# 3. Create categorical features
df['is_urban'] = (df['location'] == 'urban').astype(int)
df['is_suburban'] = (df['location'] == 'suburban').astype(int)

# 4. Create time-based features (if applicable)
df['age_category'] = pd.cut(df['age'], 
                           bins=[0, 10, 20, 30, 40, 50],
                           labels=['new', 'recent', 'mid', 'old', 'very_old'])

# Prepare features for modeling
X = df.drop('price', axis=1)
y = df['price']

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Select features
selector = SelectKBest(f_regression, k=5)
X_selected = selector.fit_transform(X_scaled, y)

# Train model
model = RandomForestRegressor(n_estimators=100)
model.fit(X_selected, y)

# Get feature importance
importances = model.feature_importances_
```

## Next Steps
1. Learn about advanced feature selection techniques
2. Study domain-specific feature engineering
3. Practice with real-world datasets
4. Learn about automated feature engineering
5. Explore feature engineering for specific types of data

## Resources
- [Feature Engineering for Machine Learning](https://www.feature-engineering-for-ml.com/)
- [Scikit-learn Feature Selection](https://scikit-learn.org/stable/modules/feature_selection.html)
- [Feature Engineering in Python](https://www.kaggle.com/learn/feature-engineering)
- [Feature Engineering for Machine Learning](https://www.amazon.com/Feature-Engineering-Machine-Learning-Principles/dp/1491953241) 