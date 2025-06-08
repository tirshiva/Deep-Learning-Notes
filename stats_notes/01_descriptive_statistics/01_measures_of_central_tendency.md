# Measures of Central Tendency

## Overview
Measures of central tendency are statistical measures that identify a single value as representative of an entire distribution. They are fundamental to understanding the "center" or "typical" value of a dataset.

## Key Concepts

### 1. Mean (Arithmetic Average)
- **Definition**: The sum of all values divided by the number of values
- **Formula**: \[ \bar{x} = \frac{\sum_{i=1}^{n} x_i}{n} \]
- **Properties**:
  - Sensitive to outliers
  - All values contribute equally
  - Used for interval and ratio data
- **Python Implementation**:
```python
import numpy as np

def calculate_mean(data):
    return np.mean(data)
```

### 2. Median
- **Definition**: The middle value in an ordered dataset
- **Calculation**:
  1. Order the data
  2. If n is odd: middle value
  3. If n is even: average of two middle values
- **Properties**:
  - Not affected by outliers
  - Better for skewed distributions
  - Used for ordinal, interval, and ratio data
- **Python Implementation**:
```python
def calculate_median(data):
    return np.median(data)
```

### 3. Mode
- **Definition**: The most frequently occurring value
- **Properties**:
  - Can be used with any type of data
  - May have multiple modes
  - Useful for categorical data
- **Python Implementation**:
```python
from scipy import stats

def calculate_mode(data):
    return stats.mode(data)[0]
```

## When to Use Each Measure

### Mean
- When the data is normally distributed
- When you need to perform further statistical calculations
- When all values are equally important

### Median
- When the data is skewed
- When there are outliers
- When you need a measure that's not affected by extreme values

### Mode
- When dealing with categorical data
- When you need to know the most common value
- When the data is discrete

## Practical Example

```python
import numpy as np
import matplotlib.pyplot as plt

# Sample data
data = [2, 3, 3, 4, 5, 6, 7, 8, 9, 10]

# Calculate measures
mean = np.mean(data)
median = np.median(data)
mode = stats.mode(data)[0]

# Visualize
plt.figure(figsize=(10, 6))
plt.hist(data, bins=10, alpha=0.7)
plt.axvline(mean, color='r', linestyle='--', label=f'Mean: {mean:.2f}')
plt.axvline(median, color='g', linestyle='--', label=f'Median: {median:.2f}')
plt.axvline(mode, color='b', linestyle='--', label=f'Mode: {mode}')
plt.legend()
plt.title('Distribution with Measures of Central Tendency')
plt.show()
```

## Common Pitfalls

1. **Using mean for skewed data**
   - Solution: Use median for skewed distributions

2. **Ignoring outliers**
   - Solution: Check for outliers and consider using median

3. **Using mode for continuous data**
   - Solution: Use mean or median for continuous data

## Applications in Machine Learning

1. **Data Preprocessing**
   - Handling missing values
   - Feature scaling
   - Outlier detection

2. **Model Evaluation**
   - Understanding prediction distributions
   - Analyzing model errors

3. **Feature Engineering**
   - Creating aggregate features
   - Normalizing data

## Exercises

1. Calculate mean, median, and mode for the following dataset:
   ```python
   data = [1, 2, 2, 3, 4, 5, 5, 5, 6, 7, 8, 9]
   ```

2. Create a visualization showing the distribution and all three measures of central tendency.

3. Analyze a real-world dataset and determine which measure of central tendency is most appropriate.

## Additional Resources

- [NumPy Documentation](https://numpy.org/doc/stable/)
- [SciPy Statistics](https://docs.scipy.org/doc/scipy/reference/stats.html)
- [Matplotlib Visualization](https://matplotlib.org/) 