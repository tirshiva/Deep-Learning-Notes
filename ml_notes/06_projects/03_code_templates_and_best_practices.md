# Code Templates and Best Practices

## Project Templates

### 1. Basic ML Project Structure
```
ml_project/
├── data/
│   ├── raw/
│   ├── processed/
│   └── external/
├── notebooks/
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   ├── make_dataset.py
│   │   └── preprocess.py
│   ├── features/
│   │   ├── __init__.py
│   │   └── build_features.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── train_model.py
│   │   └── predict_model.py
│   └── visualization/
│       ├── __init__.py
│       └── visualize.py
├── tests/
│   ├── __init__.py
│   ├── test_data.py
│   ├── test_features.py
│   └── test_models.py
├── configs/
│   ├── data_config.yaml
│   └── model_config.yaml
├── models/
├── logs/
├── docs/
├── requirements.txt
├── setup.py
└── README.md
```

### 2. Configuration Template
```yaml
# configs/model_config.yaml
data:
  raw_data_path: "data/raw/dataset.csv"
  processed_data_path: "data/processed/processed_data.csv"
  train_test_split: 0.2
  random_state: 42

features:
  categorical_features:
    - "category1"
    - "category2"
  numerical_features:
    - "feature1"
    - "feature2"
  target: "target"

model:
  type: "random_forest"
  params:
    n_estimators: 100
    max_depth: 10
    random_state: 42

training:
  batch_size: 32
  epochs: 10
  learning_rate: 0.001
  early_stopping_patience: 5

logging:
  log_dir: "logs"
  level: "INFO"
```

## Code Templates

### 1. Data Processing Pipeline
```python
# src/data/make_dataset.py
import pandas as pd
import numpy as np
from typing import Tuple, Dict
import logging
from pathlib import Path

class DataProcessor:
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def load_data(self, file_path: str) -> pd.DataFrame:
        """Load data from file"""
        self.logger.info(f"Loading data from {file_path}")
        return pd.read_csv(file_path)
    
    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess data"""
        self.logger.info("Preprocessing data")
        
        # Handle missing values
        df = self._handle_missing_values(df)
        
        # Handle categorical variables
        df = self._encode_categorical_variables(df)
        
        # Scale numerical variables
        df = self._scale_numerical_variables(df)
        
        return df
    
    def split_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Split data into train and test sets"""
        self.logger.info("Splitting data into train and test sets")
        
        X = df.drop(self.config['features']['target'], axis=1)
        y = df[self.config['features']['target']]
        
        return train_test_split(
            X, y,
            test_size=self.config['data']['train_test_split'],
            random_state=self.config['data']['random_state']
        )
    
    def save_processed_data(self, df: pd.DataFrame, file_path: str):
        """Save processed data"""
        self.logger.info(f"Saving processed data to {file_path}")
        df.to_csv(file_path, index=False)
```

### 2. Feature Engineering Pipeline
```python
# src/features/build_features.py
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import numpy as np
import pandas as pd

class FeatureEngineer:
    def __init__(self, config: Dict):
        self.config = config
        self.preprocessor = self._build_preprocessor()
    
    def _build_preprocessor(self) -> ColumnTransformer:
        """Build preprocessing pipeline"""
        # Define preprocessing steps for numerical features
        numerical_transformer = Pipeline(steps=[
            ('scaler', StandardScaler())
        ])
        
        # Define preprocessing steps for categorical features
        categorical_transformer = Pipeline(steps=[
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])
        
        # Combine preprocessing steps
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, self.config['features']['numerical_features']),
                ('cat', categorical_transformer, self.config['features']['categorical_features'])
            ]
        )
        
        return preprocessor
    
    def transform_features(self, df: pd.DataFrame) -> np.ndarray:
        """Transform features using preprocessing pipeline"""
        return self.preprocessor.fit_transform(df)
```

### 3. Model Training Pipeline
```python
# src/models/train_model.py
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
import logging
from pathlib import Path

class ModelTrainer:
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.model = self._build_model()
    
    def _build_model(self):
        """Build model based on configuration"""
        if self.config['model']['type'] == 'random_forest':
            return RandomForestClassifier(**self.config['model']['params'])
        elif self.config['model']['type'] == 'neural_network':
            return self._build_neural_network()
        else:
            raise ValueError(f"Unknown model type: {self.config['model']['type']}")
    
    def train_model(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train model"""
        self.logger.info("Training model")
        self.model.fit(X_train, y_train)
        return self.model
    
    def evaluate_model(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """Evaluate model performance"""
        self.logger.info("Evaluating model")
        
        predictions = self.model.predict(X_test)
        
        metrics = {
            'accuracy': accuracy_score(y_test, predictions),
            'precision': precision_score(y_test, predictions),
            'recall': recall_score(y_test, predictions),
            'f1': f1_score(y_test, predictions)
        }
        
        return metrics
    
    def save_model(self, model_path: str):
        """Save trained model"""
        self.logger.info(f"Saving model to {model_path}")
        joblib.dump(self.model, model_path)
```

### 4. Model Prediction Pipeline
```python
# src/models/predict_model.py
import joblib
import numpy as np
import pandas as pd
from typing import Dict, List

class ModelPredictor:
    def __init__(self, model_path: str):
        self.model = joblib.load(model_path)
    
    def predict(self, features: np.ndarray) -> np.ndarray:
        """Make predictions"""
        return self.model.predict(features)
    
    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """Make probability predictions"""
        return self.model.predict_proba(features)
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance"""
        if hasattr(self.model, 'feature_importances_'):
            return dict(zip(
                self.model.feature_names_in_,
                self.model.feature_importances_
            ))
        else:
            raise AttributeError("Model does not have feature importance")
```

## Best Practices

### 1. Code Organization
```python
# Example of well-organized code
from typing import Dict, List, Tuple
import logging
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

class MLPipeline:
    def __init__(self, config_path: str):
        self.config = self._load_config(config_path)
        self.logger = self._setup_logging()
        self.data_processor = DataProcessor(self.config)
        self.feature_engineer = FeatureEngineer(self.config)
        self.model_trainer = ModelTrainer(self.config)
    
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration"""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging"""
        logger = logging.getLogger(__name__)
        logger.setLevel(self.config['logging']['level'])
        
        # Add handlers
        handler = logging.FileHandler(
            Path(self.config['logging']['log_dir']) / 'pipeline.log'
        )
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        return logger
    
    def run_pipeline(self):
        """Run end-to-end pipeline"""
        try:
            # Load and preprocess data
            self.logger.info("Loading and preprocessing data")
            df = self.data_processor.load_data(
                self.config['data']['raw_data_path']
            )
            df = self.data_processor.preprocess_data(df)
            
            # Split data
            X_train, X_test, y_train, y_test = self.data_processor.split_data(df)
            
            # Transform features
            self.logger.info("Transforming features")
            X_train_transformed = self.feature_engineer.transform_features(X_train)
            X_test_transformed = self.feature_engineer.transform_features(X_test)
            
            # Train model
            self.logger.info("Training model")
            model = self.model_trainer.train_model(X_train_transformed, y_train)
            
            # Evaluate model
            self.logger.info("Evaluating model")
            metrics = self.model_trainer.evaluate_model(
                X_test_transformed, y_test
            )
            
            # Save model
            self.logger.info("Saving model")
            self.model_trainer.save_model(
                Path(self.config['model']['save_path'])
            )
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Pipeline failed: {str(e)}")
            raise
```

### 2. Error Handling
```python
class ErrorHandler:
    def __init__(self, logger: logging.Logger):
        self.logger = logger
    
    def handle_data_error(self, error: Exception):
        """Handle data-related errors"""
        self.logger.error(f"Data error: {str(error)}")
        # Implement specific error handling logic
    
    def handle_model_error(self, error: Exception):
        """Handle model-related errors"""
        self.logger.error(f"Model error: {str(error)}")
        # Implement specific error handling logic
    
    def handle_prediction_error(self, error: Exception):
        """Handle prediction-related errors"""
        self.logger.error(f"Prediction error: {str(error)}")
        # Implement specific error handling logic
```

### 3. Testing
```python
# tests/test_data.py
import pytest
import pandas as pd
import numpy as np
from src.data.make_dataset import DataProcessor

def test_data_loading():
    """Test data loading functionality"""
    config = {
        'data': {
            'raw_data_path': 'tests/data/test_data.csv'
        }
    }
    processor = DataProcessor(config)
    df = processor.load_data(config['data']['raw_data_path'])
    assert isinstance(df, pd.DataFrame)
    assert not df.empty

def test_data_preprocessing():
    """Test data preprocessing functionality"""
    config = {
        'features': {
            'categorical_features': ['category'],
            'numerical_features': ['value'],
            'target': 'target'
        }
    }
    processor = DataProcessor(config)
    df = pd.DataFrame({
        'category': ['A', 'B', 'A'],
        'value': [1, 2, 3],
        'target': [0, 1, 0]
    })
    processed_df = processor.preprocess_data(df)
    assert isinstance(processed_df, pd.DataFrame)
    assert not processed_df.isnull().any().any()
```

## Documentation

### 1. README Template
```markdown
# ML Project Name

## Overview
Brief description of the project and its goals.

## Installation
```bash
pip install -r requirements.txt
```

## Usage
```python
from src.pipeline import MLPipeline

pipeline = MLPipeline('configs/config.yaml')
metrics = pipeline.run_pipeline()
```

## Project Structure
```
project_root/
├── data/
├── notebooks/
├── src/
├── tests/
└── ...
```

## Configuration
Describe configuration options and how to modify them.

## Testing
```bash
pytest tests/
```

## Contributing
Guidelines for contributing to the project.

## License
Project license information.
```

### 2. Code Documentation
```python
def process_data(data: pd.DataFrame, config: Dict) -> pd.DataFrame:
    """
    Process input data according to configuration.
    
    Args:
        data (pd.DataFrame): Input data to process
        config (Dict): Configuration dictionary containing processing parameters
    
    Returns:
        pd.DataFrame: Processed data
    
    Raises:
        ValueError: If required columns are missing
        TypeError: If input data is not a DataFrame
    """
    # Implementation
```

## Resources
- [Python Best Practices](https://docs.python-guide.org/)
- [ML Project Structure](https://drivendata.github.io/cookiecutter-data-science/)
- [Testing Best Practices](https://docs.pytest.org/)
- [Documentation Best Practices](https://www.sphinx-doc.org/)
- [Code Style Guide](https://www.python.org/dev/peps/pep-0008/) 