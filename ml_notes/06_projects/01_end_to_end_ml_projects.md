# End-to-End Machine Learning Projects

## Project Structure and Best Practices

### 1. Project Organization
```
project_root/
├── data/
│   ├── raw/
│   ├── processed/
│   └── external/
├── notebooks/
├── src/
│   ├── data/
│   ├── features/
│   ├── models/
│   └── visualization/
├── tests/
├── configs/
├── models/
├── logs/
└── docs/
```

### 2. Best Practices
```python
# 1. Configuration Management
import yaml
from dataclasses import dataclass

@dataclass
class Config:
    data_path: str
    model_params: dict
    training_params: dict

def load_config(config_path: str) -> Config:
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    return Config(**config_dict)

# 2. Logging
import logging
from datetime import datetime

def setup_logging(log_dir: str):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = f"{log_dir}/run_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

# 3. Experiment Tracking
from mlflow import log_metric, log_param, log_artifacts

def track_experiment(experiment_name: str, params: dict, metrics: dict):
    with mlflow.start_run(run_name=experiment_name):
        # Log parameters
        for param_name, param_value in params.items():
            log_param(param_name, param_value)
        
        # Log metrics
        for metric_name, metric_value in metrics.items():
            log_metric(metric_name, metric_value)
```

## Real-World Applications

### 1. Customer Churn Prediction
```python
class ChurnPredictionProject:
    def __init__(self, config_path: str):
        self.config = load_config(config_path)
        self.logger = logging.getLogger(__name__)
    
    def prepare_data(self):
        """Data preparation pipeline"""
        self.logger.info("Loading and preprocessing data...")
        # Load data
        df = pd.read_csv(self.config.data_path)
        
        # Feature engineering
        df = self._engineer_features(df)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            df.drop('churn', axis=1),
            df['churn'],
            test_size=0.2,
            random_state=42
        )
        
        return X_train, X_test, y_train, y_test
    
    def train_model(self, X_train, y_train):
        """Model training pipeline"""
        self.logger.info("Training model...")
        model = RandomForestClassifier(**self.config.model_params)
        model.fit(X_train, y_train)
        return model
    
    def evaluate_model(self, model, X_test, y_test):
        """Model evaluation pipeline"""
        self.logger.info("Evaluating model...")
        predictions = model.predict(X_test)
        metrics = {
            'accuracy': accuracy_score(y_test, predictions),
            'precision': precision_score(y_test, predictions),
            'recall': recall_score(y_test, predictions),
            'f1': f1_score(y_test, predictions)
        }
        return metrics
    
    def run_pipeline(self):
        """End-to-end pipeline execution"""
        # Setup logging
        setup_logging(self.config.log_dir)
        
        # Prepare data
        X_train, X_test, y_train, y_test = self.prepare_data()
        
        # Train model
        model = self.train_model(X_train, y_train)
        
        # Evaluate model
        metrics = self.evaluate_model(model, X_test, y_test)
        
        # Track experiment
        track_experiment(
            experiment_name="churn_prediction",
            params=self.config.model_params,
            metrics=metrics
        )
        
        return model, metrics
```

### 2. Image Classification System
```python
class ImageClassificationProject:
    def __init__(self, config_path: str):
        self.config = load_config(config_path)
        self.logger = logging.getLogger(__name__)
    
    def prepare_data(self):
        """Data preparation pipeline"""
        self.logger.info("Loading and preprocessing images...")
        # Load and preprocess images
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True
        )
        
        test_datagen = ImageDataGenerator(rescale=1./255)
        
        train_generator = train_datagen.flow_from_directory(
            self.config.train_data_dir,
            target_size=(224, 224),
            batch_size=32,
            class_mode='categorical'
        )
        
        test_generator = test_datagen.flow_from_directory(
            self.config.test_data_dir,
            target_size=(224, 224),
            batch_size=32,
            class_mode='categorical'
        )
        
        return train_generator, test_generator
    
    def build_model(self):
        """Model building pipeline"""
        self.logger.info("Building model...")
        base_model = ResNet50(
            weights='imagenet',
            include_top=False,
            input_shape=(224, 224, 3)
        )
        
        model = Sequential([
            base_model,
            GlobalAveragePooling2D(),
            Dense(512, activation='relu'),
            Dropout(0.5),
            Dense(self.config.num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train_model(self, model, train_generator, test_generator):
        """Model training pipeline"""
        self.logger.info("Training model...")
        callbacks = [
            ModelCheckpoint(
                self.config.model_path,
                monitor='val_accuracy',
                save_best_only=True
            ),
            EarlyStopping(
                monitor='val_loss',
                patience=5
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=3
            )
        ]
        
        history = model.fit(
            train_generator,
            validation_data=test_generator,
            epochs=self.config.epochs,
            callbacks=callbacks
        )
        
        return history
    
    def run_pipeline(self):
        """End-to-end pipeline execution"""
        # Setup logging
        setup_logging(self.config.log_dir)
        
        # Prepare data
        train_generator, test_generator = self.prepare_data()
        
        # Build model
        model = self.build_model()
        
        # Train model
        history = self.train_model(model, train_generator, test_generator)
        
        # Track experiment
        track_experiment(
            experiment_name="image_classification",
            params=self.config.model_params,
            metrics=history.history
        )
        
        return model, history
```

### 3. Natural Language Processing System
```python
class NLPProject:
    def __init__(self, config_path: str):
        self.config = load_config(config_path)
        self.logger = logging.getLogger(__name__)
    
    def prepare_data(self):
        """Data preparation pipeline"""
        self.logger.info("Loading and preprocessing text data...")
        # Load data
        df = pd.read_csv(self.config.data_path)
        
        # Text preprocessing
        df['processed_text'] = df['text'].apply(self._preprocess_text)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            df['processed_text'],
            df['label'],
            test_size=0.2,
            random_state=42
        )
        
        # Tokenize
        tokenizer = Tokenizer(num_words=self.config.max_words)
        tokenizer.fit_on_texts(X_train)
        
        X_train_seq = tokenizer.texts_to_sequences(X_train)
        X_test_seq = tokenizer.texts_to_sequences(X_test)
        
        # Pad sequences
        X_train_pad = pad_sequences(X_train_seq, maxlen=self.config.max_len)
        X_test_pad = pad_sequences(X_test_seq, maxlen=self.config.max_len)
        
        return X_train_pad, X_test_pad, y_train, y_test, tokenizer
    
    def build_model(self, tokenizer):
        """Model building pipeline"""
        self.logger.info("Building model...")
        model = Sequential([
            Embedding(
                len(tokenizer.word_index) + 1,
                self.config.embedding_dim,
                input_length=self.config.max_len
            ),
            Bidirectional(LSTM(64, return_sequences=True)),
            Bidirectional(LSTM(32)),
            Dense(64, activation='relu'),
            Dropout(0.5),
            Dense(self.config.num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def run_pipeline(self):
        """End-to-end pipeline execution"""
        # Setup logging
        setup_logging(self.config.log_dir)
        
        # Prepare data
        X_train, X_test, y_train, y_test, tokenizer = self.prepare_data()
        
        # Build model
        model = self.build_model(tokenizer)
        
        # Train model
        history = model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=self.config.epochs,
            batch_size=self.config.batch_size,
            callbacks=[
                EarlyStopping(patience=3),
                ReduceLROnPlateau(factor=0.2, patience=2)
            ]
        )
        
        # Track experiment
        track_experiment(
            experiment_name="nlp_classification",
            params=self.config.model_params,
            metrics=history.history
        )
        
        return model, history
```

## Code Templates

### 1. Project Configuration
```yaml
# config.yaml
data:
  path: "data/raw/dataset.csv"
  train_test_split: 0.2
  random_state: 42

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

logging:
  log_dir: "logs"
  level: "INFO"
```

### 2. Data Pipeline
```python
class DataPipeline:
    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def load_data(self):
        """Load data from source"""
        self.logger.info("Loading data...")
        return pd.read_csv(self.config.data_path)
    
    def preprocess_data(self, df: pd.DataFrame):
        """Preprocess data"""
        self.logger.info("Preprocessing data...")
        # Handle missing values
        df = self._handle_missing_values(df)
        
        # Feature engineering
        df = self._engineer_features(df)
        
        # Feature scaling
        df = self._scale_features(df)
        
        return df
    
    def split_data(self, df: pd.DataFrame):
        """Split data into train and test sets"""
        self.logger.info("Splitting data...")
        X = df.drop('target', axis=1)
        y = df['target']
        
        return train_test_split(
            X, y,
            test_size=self.config.train_test_split,
            random_state=self.config.random_state
        )
```

### 3. Model Pipeline
```python
class ModelPipeline:
    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def build_model(self):
        """Build model based on configuration"""
        self.logger.info("Building model...")
        if self.config.model_type == "random_forest":
            return RandomForestClassifier(**self.config.model_params)
        elif self.config.model_type == "neural_network":
            return self._build_neural_network()
        else:
            raise ValueError(f"Unknown model type: {self.config.model_type}")
    
    def train_model(self, model, X_train, y_train):
        """Train model"""
        self.logger.info("Training model...")
        model.fit(X_train, y_train)
        return model
    
    def evaluate_model(self, model, X_test, y_test):
        """Evaluate model"""
        self.logger.info("Evaluating model...")
        predictions = model.predict(X_test)
        metrics = {
            'accuracy': accuracy_score(y_test, predictions),
            'precision': precision_score(y_test, predictions),
            'recall': recall_score(y_test, predictions),
            'f1': f1_score(y_test, predictions)
        }
        return metrics
```

## Best Practices Checklist

1. **Project Setup**
   - [ ] Use version control (Git)
   - [ ] Create virtual environment
   - [ ] Set up project structure
   - [ ] Create requirements.txt
   - [ ] Add README.md

2. **Data Management**
   - [ ] Document data sources
   - [ ] Implement data versioning
   - [ ] Create data validation
   - [ ] Set up data pipeline
   - [ ] Implement data preprocessing

3. **Model Development**
   - [ ] Use configuration files
   - [ ] Implement logging
   - [ ] Set up experiment tracking
   - [ ] Create model pipeline
   - [ ] Implement model evaluation

4. **Testing**
   - [ ] Write unit tests
   - [ ] Create integration tests
   - [ ] Implement CI/CD
   - [ ] Add code coverage
   - [ ] Document test cases

5. **Deployment**
   - [ ] Containerize application
   - [ ] Set up monitoring
   - [ ] Implement error handling
   - [ ] Create API documentation
   - [ ] Set up backup system

## Resources
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [DVC Documentation](https://dvc.org/doc)
- [Cookiecutter Data Science](https://drivendata.github.io/cookiecutter-data-science/)
- [MLOps Best Practices](https://www.mlops.org/)
- [Model Deployment Guide](https://www.tensorflow.org/tfx/guide) 