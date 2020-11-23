import pandas as pd 
import numpy as np
from matplotlib import pyplot as plt


# y = mx + b

# Step 1 Generate random variables
np.random.seed(0)

# Step 2 define x
x: np.ndarray = 2.5 * np.random.randn(100) + 1.5

# Step 3 define b (y-intercept)
b: np.ndarray = 0.5 * np.random.randn(100)

# Step 4 define m (slope)
m: float = 0.3

# Step 5 find y actual values
y = m*x + b

# Step 6 Place it data frame
df_lr: pd.DataFrame = pd.DataFrame({"X": x, "Y": y})

# Step 7 Find the mean values of x and y
lr_mean_x = np.mean(x)
lr_mean_y = np.mean(y)

# Step 8 Make prediction using least square method formula
m_numerator = (df_lr["X"] - lr_mean_y) * (df_lr["Y"] - lr_mean_x)
m_denominator = (df_lr["X"] - lr_mean_x) ** 2
m = m_numerator.sum() / m_denominator.sum()
b = lr_mean_y - m * lr_mean_x
y_pred = m*x + b



# Step 9 plot regression against actual data
plt.figure(figsize=(12, 6))
plt.plot(x, y_pred)     # regression line
plt.plot(x, y, 'ro')   # scatter plot showing actual data
plt.title('Actual vs Predicted')
plt.xlabel('X')
plt.ylabel('y')
plt.show()
