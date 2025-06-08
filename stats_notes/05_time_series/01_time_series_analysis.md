# Time Series Analysis

## What is Time Series Analysis?

Time series analysis is a statistical method used to analyze and model data points collected over time. It helps understand patterns, trends, and seasonal variations in data, making it essential for forecasting and decision-making in machine learning applications.

## Why is Time Series Analysis Important in Machine Learning?

1. **Forecasting**
   - Predicting future values
   - Trend analysis
   - Seasonal pattern detection

2. **Pattern Recognition**
   - Identifying cycles
   - Detecting anomalies
   - Understanding dependencies

3. **Decision Making**
   - Resource planning
   - Risk assessment
   - Performance monitoring

## How Does Time Series Analysis Work?

### 1. Time Series Components

#### What are Time Series Components?
**Definition**: Basic elements that make up a time series
**Components**:
- Trend: Long-term movement
- Seasonality: Regular patterns
- Cyclical: Long-term oscillations
- Random: Irregular fluctuations

```python
def decompose_time_series(data, period=12):
    # Calculate trend using moving average
    trend = np.convolve(data, np.ones(period)/period, mode='valid')
    
    # Calculate seasonal component
    seasonal = data - trend
    
    # Calculate random component
    random = data - trend - seasonal
    
    return trend, seasonal, random

# Example
t = np.linspace(0, 10, 100)
data = np.sin(t) + 0.1 * np.random.randn(100)
trend, seasonal, random = decompose_time_series(data)
```

### 2. Stationarity

#### What is Stationarity?
**Definition**: Statistical properties remain constant over time
**Properties**:
- Constant mean
- Constant variance
- Constant autocorrelation

```python
def check_stationarity(data):
    # Calculate rolling statistics
    window = 12
    rolling_mean = np.convolve(data, np.ones(window)/window, mode='valid')
    rolling_std = np.array([np.std(data[i:i+window]) 
                           for i in range(len(data)-window+1)])
    
    # Perform Augmented Dickey-Fuller test
    from statsmodels.tsa.stattools import adfuller
    result = adfuller(data)
    
    return {
        'rolling_mean': rolling_mean,
        'rolling_std': rolling_std,
        'adf_statistic': result[0],
        'p_value': result[1]
    }

# Example
data = np.random.randn(100)
stationarity = check_stationarity(data)
```

### 3. Autocorrelation

#### What is Autocorrelation?
**Definition**: Correlation between observations at different time points
**Formula**: \[ r_k = \frac{\sum_{t=k+1}^n (y_t - \bar{y})(y_{t-k} - \bar{y})}{\sum_{t=1}^n (y_t - \bar{y})^2} \]

```python
def calculate_autocorrelation(data, max_lag=20):
    n = len(data)
    mean = np.mean(data)
    var = np.var(data)
    
    acf = []
    for lag in range(max_lag + 1):
        numerator = np.sum((data[lag:] - mean) * (data[:n-lag] - mean))
        denominator = n * var
        acf.append(numerator / denominator)
    
    return np.array(acf)

# Example
data = np.random.randn(100)
acf = calculate_autocorrelation(data)
```

## Time Series Models

### 1. ARIMA Model

#### What is ARIMA?
**Definition**: AutoRegressive Integrated Moving Average
**Components**:
- AR: AutoRegressive
- I: Integrated
- MA: Moving Average

```python
def fit_arima(data, p=1, d=1, q=1):
    from statsmodels.tsa.arima.model import ARIMA
    
    model = ARIMA(data, order=(p, d, q))
    results = model.fit()
    
    return results

# Example
data = np.random.randn(100)
model = fit_arima(data)
```

### 2. Exponential Smoothing

#### What is Exponential Smoothing?
**Definition**: Weighted average of past observations
**Formula**: \[ \hat{y}_t = \alpha y_t + (1-\alpha)\hat{y}_{t-1} \]

```python
def exponential_smoothing(data, alpha=0.3):
    smoothed = np.zeros_like(data)
    smoothed[0] = data[0]
    
    for t in range(1, len(data)):
        smoothed[t] = alpha * data[t] + (1 - alpha) * smoothed[t-1]
    
    return smoothed

# Example
data = np.random.randn(100)
smoothed = exponential_smoothing(data)
```

### 3. Seasonal Decomposition

```python
def seasonal_decomposition(data, period=12):
    from statsmodels.tsa.seasonal import seasonal_decompose
    
    decomposition = seasonal_decompose(data, period=period)
    
    return {
        'trend': decomposition.trend,
        'seasonal': decomposition.seasonal,
        'resid': decomposition.resid
    }

# Example
data = np.random.randn(100)
decomposition = seasonal_decomposition(data)
```

## Visualizing Time Series

```python
import matplotlib.pyplot as plt
import seaborn as sns

def plot_time_series(data, title='Time Series Plot'):
    plt.figure(figsize=(12, 6))
    plt.plot(data)
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.show()

def plot_autocorrelation(acf, title='Autocorrelation Function'):
    plt.figure(figsize=(10, 6))
    plt.stem(range(len(acf)), acf)
    plt.title(title)
    plt.xlabel('Lag')
    plt.ylabel('Autocorrelation')
    plt.show()

def plot_decomposition(trend, seasonal, random, title='Time Series Decomposition'):
    plt.figure(figsize=(12, 8))
    
    plt.subplot(311)
    plt.plot(trend)
    plt.title('Trend')
    
    plt.subplot(312)
    plt.plot(seasonal)
    plt.title('Seasonal')
    
    plt.subplot(313)
    plt.plot(random)
    plt.title('Random')
    
    plt.tight_layout()
    plt.show()

# Example
t = np.linspace(0, 10, 100)
data = np.sin(t) + 0.1 * np.random.randn(100)
plot_time_series(data)
```

## Common Pitfalls

1. **Non-stationarity**
   - Problem: Changing statistical properties
   - Solution: Differencing or transformation

2. **Missing Values**
   - Problem: Gaps in time series
   - Solution: Interpolation or imputation

3. **Seasonality**
   - Problem: Complex seasonal patterns
   - Solution: Seasonal decomposition

## Applications in Machine Learning

### 1. Forecasting
```python
def forecast_arima(model, steps=10):
    return model.forecast(steps=steps)

# Example
data = np.random.randn(100)
model = fit_arima(data)
forecast = forecast_arima(model)
```

### 2. Anomaly Detection
```python
def detect_anomalies(data, threshold=2):
    mean = np.mean(data)
    std = np.std(data)
    z_scores = np.abs((data - mean) / std)
    return z_scores > threshold

# Example
data = np.random.randn(100)
anomalies = detect_anomalies(data)
```

### 3. Feature Engineering
```python
def create_time_features(data):
    return {
        'lag_1': np.roll(data, 1),
        'lag_2': np.roll(data, 2),
        'rolling_mean': np.convolve(data, np.ones(12)/12, mode='valid'),
        'rolling_std': np.array([np.std(data[i:i+12]) 
                                for i in range(len(data)-11)])
    }

# Example
data = np.random.randn(100)
features = create_time_features(data)
```

## Interview Questions

1. **Basic Concepts**
   - What's the difference between trend and seasonality?
   - How do you handle non-stationary time series?
   - What's the role of autocorrelation in time series analysis?

2. **Practical Applications**
   - How do you choose between different time series models?
   - What's the role of cross-validation in time series?
   - How do you handle missing values in time series data?

## Exercises

1. Implement different time series models:
   ```python
   # a) SARIMA
   # b) Prophet
   # c) LSTM
   ```

2. Create a function for time series cross-validation:
   ```python
   def time_series_cv(data, model, n_splits=5):
       n = len(data)
       split_size = n // n_splits
       
       scores = []
       for i in range(n_splits):
           train_end = n - (n_splits - i) * split_size
           test_end = train_end + split_size
           
           train_data = data[:train_end]
           test_data = data[train_end:test_end]
           
           model.fit(train_data)
           predictions = model.predict(len(test_data))
           score = np.mean((test_data - predictions) ** 2)
           scores.append(score)
       
       return np.mean(scores)
   ```

3. Implement a function for seasonal adjustment:
   ```python
   def seasonal_adjustment(data, period=12):
       decomposition = seasonal_decomposition(data, period)
       return data - decomposition.seasonal
   ```

## Additional Resources

- [StatsModels Time Series](https://www.statsmodels.org/stable/tsa.html)
- [Prophet Documentation](https://facebook.github.io/prophet/)
- [Time Series Analysis in Python](https://www.statsmodels.org/stable/examples/index.html#time-series-analysis) 