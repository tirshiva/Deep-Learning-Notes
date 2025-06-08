# Time Series Analysis

## Background and Introduction
Time series analysis is a specialized branch of machine learning that deals with data points collected or recorded at specific time intervals. It's widely used in various domains such as finance, weather forecasting, sales prediction, and IoT data analysis. Understanding time series data requires special techniques due to its temporal nature and potential dependencies between observations.

## What is Time Series Analysis?
Time series analysis involves studying data points collected over time to:
1. Understand underlying patterns and trends
2. Make predictions about future values
3. Identify seasonal patterns and cycles
4. Detect anomalies and outliers
5. Understand relationships between variables over time

## Why Time Series Analysis?
1. **Temporal Dependencies**: Data points are often correlated with previous values
2. **Seasonality**: Many phenomena show repeating patterns
3. **Trend Analysis**: Helps understand long-term changes
4. **Forecasting**: Enables future value prediction
5. **Anomaly Detection**: Identifies unusual patterns

## How to Analyze Time Series Data?

### 1. Data Preprocessing and Visualization
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm

def preprocess_time_series(data, date_column, value_column):
    # Convert to datetime
    data[date_column] = pd.to_datetime(data[date_column])
    data.set_index(date_column, inplace=True)
    
    # Handle missing values
    data = data.fillna(method='ffill')
    
    # Remove outliers using IQR method
    Q1 = data[value_column].quantile(0.25)
    Q3 = data[value_column].quantile(0.75)
    IQR = Q3 - Q1
    data = data[
        (data[value_column] >= Q1 - 1.5 * IQR) &
        (data[value_column] <= Q3 + 1.5 * IQR)
    ]
    
    return data

def visualize_time_series(data, value_column):
    # Create figure with subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 12))
    
    # Plot original time series
    data[value_column].plot(ax=ax1)
    ax1.set_title('Original Time Series')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Value')
    
    # Plot rolling mean and std
    rolling_mean = data[value_column].rolling(window=12).mean()
    rolling_std = data[value_column].rolling(window=12).std()
    data[value_column].plot(ax=ax2, label='Original')
    rolling_mean.plot(ax=ax2, label='Rolling Mean')
    rolling_std.plot(ax=ax2, label='Rolling Std')
    ax2.set_title('Rolling Statistics')
    ax2.legend()
    
    # Plot seasonal decomposition
    decomposition = seasonal_decompose(data[value_column], period=12)
    decomposition.plot(ax=ax3)
    ax3.set_title('Seasonal Decomposition')
    
    plt.tight_layout()
    plt.show()
    
    return decomposition

# Example usage
def demonstrate_preprocessing():
    # Create sample time series data
    dates = pd.date_range(start='2020-01-01', end='2022-12-31', freq='M')
    values = np.random.normal(0, 1, len(dates)).cumsum() + 100
    data = pd.DataFrame({
        'date': dates,
        'value': values
    })
    
    # Preprocess and visualize
    processed_data = preprocess_time_series(data, 'date', 'value')
    decomposition = visualize_time_series(processed_data, 'value')
    
    return processed_data, decomposition
```

### 2. Stationarity Testing and Transformation
```python
def test_stationarity(data, value_column):
    # Perform Augmented Dickey-Fuller test
    result = adfuller(data[value_column])
    
    print('Augmented Dickey-Fuller Test:')
    print(f'ADF Statistic: {result[0]}')
    print(f'p-value: {result[1]}')
    print('Critical Values:')
    for key, value in result[4].items():
        print(f'\t{key}: {value}')
    
    return result

def make_stationary(data, value_column):
    # Calculate first difference
    data['diff'] = data[value_column].diff()
    
    # Calculate seasonal difference
    data['seasonal_diff'] = data[value_column].diff(12)
    
    # Plot differences
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 8))
    data['diff'].plot(ax=ax1)
    ax1.set_title('First Difference')
    data['seasonal_diff'].plot(ax=ax2)
    ax2.set_title('Seasonal Difference')
    plt.tight_layout()
    plt.show()
    
    return data

# Example usage
def demonstrate_stationarity():
    # Get processed data
    processed_data, _ = demonstrate_preprocessing()
    
    # Test stationarity
    result = test_stationarity(processed_data, 'value')
    
    # Make stationary
    stationary_data = make_stationary(processed_data, 'value')
    
    return stationary_data, result
```

### 3. ARIMA Modeling
```python
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

def fit_arima(data, value_column, order):
    # Fit ARIMA model
    model = ARIMA(data[value_column], order=order)
    results = model.fit()
    
    # Print model summary
    print(results.summary())
    
    # Plot diagnostics
    results.plot_diagnostics(figsize=(15, 12))
    plt.show()
    
    return results

def evaluate_arima(model, test_data, value_column):
    # Make predictions
    predictions = model.forecast(steps=len(test_data))
    
    # Calculate metrics
    mse = mean_squared_error(test_data[value_column], predictions)
    mae = mean_absolute_error(test_data[value_column], predictions)
    
    # Plot predictions
    plt.figure(figsize=(15, 6))
    plt.plot(test_data.index, test_data[value_column], label='Actual')
    plt.plot(test_data.index, predictions, label='Predicted')
    plt.title('ARIMA Predictions vs Actual')
    plt.legend()
    plt.show()
    
    return predictions, mse, mae

# Example usage
def demonstrate_arima():
    # Get stationary data
    stationary_data, _ = demonstrate_stationarity()
    
    # Split data
    train_size = int(len(stationary_data) * 0.8)
    train_data = stationary_data[:train_size]
    test_data = stationary_data[train_size:]
    
    # Fit ARIMA model
    model = fit_arima(train_data, 'value', order=(1, 1, 1))
    
    # Evaluate model
    predictions, mse, mae = evaluate_arima(model, test_data, 'value')
    
    return model, predictions, mse, mae
```

### 4. Prophet Modeling
```python
from prophet import Prophet

def fit_prophet(data, date_column, value_column):
    # Prepare data for Prophet
    prophet_data = pd.DataFrame({
        'ds': data.index,
        'y': data[value_column]
    })
    
    # Fit Prophet model
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=True
    )
    model.fit(prophet_data)
    
    # Make future predictions
    future = model.make_future_dataframe(periods=365)
    forecast = model.predict(future)
    
    # Plot forecast
    fig = model.plot(forecast)
    plt.title('Prophet Forecast')
    plt.show()
    
    # Plot components
    fig = model.plot_components(forecast)
    plt.show()
    
    return model, forecast

# Example usage
def demonstrate_prophet():
    # Get processed data
    processed_data, _ = demonstrate_preprocessing()
    
    # Fit Prophet model
    model, forecast = fit_prophet(processed_data, 'date', 'value')
    
    return model, forecast
```

## Model Evaluation

### 1. Time Series Cross-Validation
```python
from sklearn.model_selection import TimeSeriesSplit

def time_series_cv(data, value_column, model_type='arima'):
    # Create time series cross-validation splits
    tscv = TimeSeriesSplit(n_splits=5)
    
    # Initialize metrics
    mse_scores = []
    mae_scores = []
    
    for train_idx, test_idx in tscv.split(data):
        train_data = data.iloc[train_idx]
        test_data = data.iloc[test_idx]
        
        if model_type == 'arima':
            # Fit ARIMA model
            model = ARIMA(train_data[value_column], order=(1, 1, 1))
            results = model.fit()
            predictions = results.forecast(steps=len(test_data))
        else:
            # Fit Prophet model
            prophet_data = pd.DataFrame({
                'ds': train_data.index,
                'y': train_data[value_column]
            })
            model = Prophet()
            model.fit(prophet_data)
            future = model.make_future_dataframe(periods=len(test_data))
            forecast = model.predict(future)
            predictions = forecast['yhat'].iloc[-len(test_data):]
        
        # Calculate metrics
        mse = mean_squared_error(test_data[value_column], predictions)
        mae = mean_absolute_error(test_data[value_column], predictions)
        
        mse_scores.append(mse)
        mae_scores.append(mae)
    
    return np.mean(mse_scores), np.mean(mae_scores)
```

### 2. Model Comparison
```python
def compare_models(data, value_column):
    # Get predictions from both models
    _, arima_pred, arima_mse, arima_mae = demonstrate_arima()
    _, prophet_forecast = demonstrate_prophet()
    
    # Plot comparison
    plt.figure(figsize=(15, 6))
    plt.plot(data.index[-len(arima_pred):], data[value_column].iloc[-len(arima_pred):],
             label='Actual')
    plt.plot(data.index[-len(arima_pred):], arima_pred, label='ARIMA')
    plt.plot(data.index[-len(arima_pred):], prophet_forecast['yhat'].iloc[-len(arima_pred):],
             label='Prophet')
    plt.title('Model Comparison')
    plt.legend()
    plt.show()
    
    # Print metrics
    print('ARIMA Metrics:')
    print(f'MSE: {arima_mse:.4f}')
    print(f'MAE: {arima_mae:.4f}')
    
    prophet_mse = mean_squared_error(
        data[value_column].iloc[-len(arima_pred):],
        prophet_forecast['yhat'].iloc[-len(arima_pred):]
    )
    prophet_mae = mean_absolute_error(
        data[value_column].iloc[-len(arima_pred):],
        prophet_forecast['yhat'].iloc[-len(arima_pred):]
    )
    
    print('\nProphet Metrics:')
    print(f'MSE: {prophet_mse:.4f}')
    print(f'MAE: {prophet_mae:.4f}')
```

## Common Interview Questions

1. **Q: What is stationarity in time series analysis?**
   - A: A stationary time series has constant mean, variance, and autocorrelation over time. Stationarity is important because many time series models assume stationarity. We can test for stationarity using the Augmented Dickey-Fuller test and make non-stationary data stationary through differencing or transformation.

2. **Q: What is the difference between ARIMA and Prophet?**
   - A: ARIMA is a traditional statistical model that handles trend and seasonality through differencing, while Prophet is a more modern approach that explicitly models trend, seasonality, and holidays. Prophet is more robust to missing data and can handle multiple seasonality patterns, but ARIMA might be more suitable for short-term forecasting.

3. **Q: How do you handle seasonality in time series data?**
   - A: Seasonality can be handled through:
     - Seasonal differencing
     - Seasonal decomposition
     - Seasonal ARIMA (SARIMA)
     - Prophet's built-in seasonality components
     - Fourier terms for complex seasonality

## Hands-on Task: Sales Forecasting

### Project: Retail Sales Prediction
```python
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

# Create sample retail sales data
np.random.seed(42)
dates = pd.date_range(start='2018-01-01', end='2022-12-31', freq='D')
base_sales = 1000
trend = np.linspace(0, 500, len(dates))
seasonality = 100 * np.sin(np.linspace(0, 4*np.pi, len(dates)))
noise = np.random.normal(0, 50, len(dates))
sales = base_sales + trend + seasonality + noise

# Create DataFrame
data = pd.DataFrame({
    'date': dates,
    'sales': sales
})

# Preprocess data
processed_data = preprocess_time_series(data, 'date', 'sales')
decomposition = visualize_time_series(processed_data, 'sales')

# Test stationarity
stationary_data, stationarity_result = demonstrate_stationarity()

# Fit and evaluate ARIMA model
arima_model, arima_pred, arima_mse, arima_mae = demonstrate_arima()

# Fit and evaluate Prophet model
prophet_model, prophet_forecast = demonstrate_prophet()

# Compare models
compare_models(processed_data, 'sales')

# Perform time series cross-validation
mse_cv, mae_cv = time_series_cv(processed_data, 'sales')
print(f'\nCross-validation Metrics:')
print(f'MSE: {mse_cv:.4f}')
print(f'MAE: {mae_cv:.4f}')
```

## Next Steps
1. Learn about advanced time series models (SARIMA, GARCH)
2. Study multivariate time series analysis
3. Explore deep learning approaches (LSTM, GRU)
4. Practice with real-world datasets
5. Learn about time series anomaly detection

## Resources
- [Statsmodels Documentation](https://www.statsmodels.org/)
- [Prophet Documentation](https://facebook.github.io/prophet/)
- [Time Series Analysis: Forecasting and Control](https://www.amazon.com/Time-Analysis-Forecasting-Control-Box/dp/1118675029)
- [Forecasting: Principles and Practice](https://otexts.com/fpp3/) 