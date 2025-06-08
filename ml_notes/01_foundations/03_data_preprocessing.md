# Data Preprocessing

## Background and Introduction
Data preprocessing is a crucial step in the machine learning pipeline that transforms raw data into a format suitable for analysis and modeling. It's often said that data scientists spend 80% of their time on data preprocessing and only 20% on actual modeling.

## What is Data Preprocessing?
Data preprocessing involves a series of steps to clean, transform, and prepare data for machine learning algorithms. It ensures that the data is:
- Clean and consistent
- In the right format
- Free from errors and outliers
- Properly scaled and normalized
- Ready for feature engineering

## Why Data Preprocessing?
1. **Data Quality**: Real-world data is often messy and incomplete
2. **Model Performance**: Clean data leads to better model performance
3. **Algorithm Requirements**: Many algorithms have specific data requirements
4. **Error Prevention**: Prevents errors during model training
5. **Insight Generation**: Helps in understanding data patterns

## How to Preprocess Data?

### 1. Data Cleaning

#### Handling Missing Values
```python
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer

# Create sample data with missing values
data = {
    'age': [25, np.nan, 30, 35, np.nan],
    'salary': [50000, 60000, np.nan, 70000, 80000],
    'experience': [2, 3, 4, np.nan, 6]
}
df = pd.DataFrame(data)

# Method 1: Remove rows with missing values
df_dropped = df.dropna()

# Method 2: Fill with mean
imputer = SimpleImputer(strategy='mean')
df_imputed = pd.DataFrame(
    imputer.fit_transform(df),
    columns=df.columns
)

# Method 3: Forward fill
df_ffill = df.fillna(method='ffill')

# Method 4: Backward fill
df_bfill = df.fillna(method='bfill')
```

#### Handling Outliers
```python
import numpy as np
from scipy import stats

# Create sample data with outliers
data = np.array([1, 2, 2, 3, 3, 3, 4, 4, 5, 100])

# Method 1: Z-score
z_scores = stats.zscore(data)
outliers = (np.abs(z_scores) > 3)
cleaned_data = data[~outliers]

# Method 2: IQR
Q1 = np.percentile(data, 25)
Q3 = np.percentile(data, 75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
cleaned_data = data[(data >= lower_bound) & (data <= upper_bound)]
```

### 2. Data Transformation

#### Scaling and Normalization
```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

# Create sample data
data = np.array([[1, 2], [3, 4], [5, 6]])

# Standardization (Z-score)
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# Min-Max Scaling
minmax_scaler = MinMaxScaler()
data_minmax = minmax_scaler.fit_transform(data)

# Robust Scaling
robust_scaler = RobustScaler()
data_robust = robust_scaler.fit_transform(data)
```

#### Encoding Categorical Variables
```python
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# Create sample data
categories = ['red', 'blue', 'green', 'red', 'blue']

# Label Encoding
le = LabelEncoder()
labels = le.fit_transform(categories)

# One-Hot Encoding
ohe = OneHotEncoder(sparse=False)
one_hot = ohe.fit_transform(np.array(categories).reshape(-1, 1))

# Using pandas get_dummies
df = pd.DataFrame({'color': categories})
df_encoded = pd.get_dummies(df, columns=['color'])
```

### 3. Feature Engineering

#### Creating New Features
```python
# Create sample data
df = pd.DataFrame({
    'date': pd.date_range('2023-01-01', periods=5),
    'value': [10, 20, 30, 40, 50]
})

# Extract date features
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day

# Create interaction features
df['value_squared'] = df['value'] ** 2
df['value_log'] = np.log(df['value'])

# Create rolling features
df['value_rolling_mean'] = df['value'].rolling(window=2).mean()
```

#### Feature Selection
```python
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier

# Create sample data
X = np.random.rand(100, 10)
y = np.random.randint(0, 2, 100)

# Method 1: Statistical tests
selector = SelectKBest(f_classif, k=5)
X_selected = selector.fit_transform(X, y)

# Method 2: Feature importance
model = RandomForestClassifier()
model.fit(X, y)
importances = model.feature_importances_
```

## Common Interview Questions

1. **Q: What are the different methods to handle missing values?**
   - A: Common methods include:
     - Removing rows/columns with missing values
     - Mean/median/mode imputation
     - Forward/backward fill
     - Interpolation
     - Advanced imputation methods (KNN, MICE)

2. **Q: When should you use standardization vs. normalization?**
   - A: Use standardization when:
     - Data follows normal distribution
     - Features have different scales
     - Using algorithms that assume normal distribution
     Use normalization when:
     - Data doesn't follow normal distribution
     - Need values in specific range (e.g., [0,1])
     - Using algorithms that don't assume normal distribution

3. **Q: How do you handle categorical variables in machine learning?**
   - A: Common approaches include:
     - Label encoding for ordinal variables
     - One-hot encoding for nominal variables
     - Target encoding for high-cardinality variables
     - Binary encoding for large categorical variables

## Hands-on Task: Complete Data Preprocessing Pipeline

### Project: Customer Churn Prediction
```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Create sample customer data
np.random.seed(42)
n_customers = 1000
data = {
    'age': np.random.normal(35, 10, n_customers),
    'tenure': np.random.normal(24, 12, n_customers),
    'monthly_charges': np.random.normal(50, 20, n_customers),
    'contract_type': np.random.choice(['monthly', 'yearly', 'two-year'], n_customers),
    'payment_method': np.random.choice(['credit', 'debit', 'bank transfer'], n_customers),
    'churn': np.random.randint(0, 2, n_customers)
}
df = pd.DataFrame(data)

# Split features and target
X = df.drop('churn', axis=1)
y = df['churn']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Define preprocessing steps
numeric_features = ['age', 'tenure', 'monthly_charges']
categorical_features = ['contract_type', 'payment_method']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(), categorical_features)
    ])

# Create pipeline
pipeline = Pipeline([
    ('preprocessor', preprocessor)
])

# Fit and transform data
X_train_processed = pipeline.fit_transform(X_train)
X_test_processed = pipeline.transform(X_test)

print("Original shape:", X_train.shape)
print("Processed shape:", X_train_processed.shape)
```

## Next Steps
1. Learn about feature engineering techniques
2. Study different scaling methods
3. Practice with real-world datasets
4. Learn about advanced preprocessing techniques
5. Explore automated preprocessing tools

## Resources
- [Scikit-learn Preprocessing Documentation](https://scikit-learn.org/stable/modules/preprocessing.html)
- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [Feature Engineering for Machine Learning](https://www.feature-engineering-for-ml.com/)
- [Data Preprocessing in Python](https://www.kaggle.com/learn/data-preprocessing) 