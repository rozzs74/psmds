import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

PATH = "./synth_temp.csv"

df: pd.DataFrame = pd.read_csv(PATH)
df = df.loc[df.Year > 1901]
df_group_year = df.groupby('Year').agg(np.mean)
window = 10
rolling = df_group_year.AverageTemperature.rolling(window).mean();
df_group_year['Year'] = df_group_year.index
x = np.ones((2, len(df_group_year)))
x[0,:] = df_group_year.Year
x[1,:] = 1
x /= x.max()
x[:,:5]


# Gradient Descent
# Step 1 set theta to be random and close to 0 and 2D array
Theta: np.ndarray = np.random.rand(2).reshape((1, 2)) * 0.1
print(Theta)
np.random.seed(255) # Ensure the same starting random values

# Step 2 predict each of the samples in the training model using random values of theta and flatten the output in 1 dimensional array
h_x: np.ndarray = np.dot(Theta, x).flatten() # y_pred and flattened

# Step 3 get the actual values of dependent variable (y-axis)
y_t: pd.DataFrame = df_group_year.AverageTemperature.values # y_actual and flattened

# Step 4 Compute mean square errors (j(theta))
j_0 = np.mean((h_x - y_t) ** 2)
# Step 5 define learning rate
# l_r: float= 0.0001
l_r: float= 1e-6

# Step 6 define number of iterations to perform gradient descent
max_epoch: int = 100000

# Step 7 Update gradient descent parameters by including learning rate this is initial view because it will be using inside iteration
Theta += l_r * np.sum((y_t - h_x ) * x, axis=1)
h_x: np.ndarray = np.dot(Theta, x).flatten()
j_0  = np.mean((h_x - y_t) ** 2)

# Step 8 Repeat the process in iteration with fixed number the stepper is 10
i: int = 0
error_hist: list = []
epoch_hist: list = []

while i <= max_epoch:
    Theta += l_r * np.sum((y_t - h_x ) * x, axis=1)
    h_x: np.ndarray = np.dot(Theta, x).flatten()
    if (i % 10) == 0:
        j_0 = np.mean((h_x - y_t) ** 2)
        error_hist.append(j_0)
        epoch_hist.append(i)
    i += 1
    if i == max_epoch:
        break

# Step 9 Compare training history and errors 
plt.figure(figsize=(10, 7))
plt.plot(epoch_hist, error_hist);
plt.title('Training History');
plt.xlabel('epoch');
plt.ylabel('Error');
plt.show()

# Step 10 Compute r-squared between actual and predicted values
r_squared = r2_score(y_t, h_x)
print(r_squared)

# Step 11 Create trend by comparing the predicted values in Gradient Descent process with actual values.
x = np.linspace(df_group_year['Year'].min(), df_group_year['Year'].max(), 20)
trend_x = np.ones((2, len(x)))
trend_x[0,:] = x
trend_x[1,:] = 1
trend_x /= trend_x.max()
h_x = np.dot(Theta, trend_x).flatten()

# Step 12 Plot the trend
fig = plt.figure(figsize=(10, 7))
ax = fig.add_axes([1, 1, 1, 1]);
ax.scatter(df_group_year.index, df_group_year.AverageTemperature, label='Raw Data', c='k');
ax.plot(df_group_year.index, rolling, c='k', linestyle='--', label=f'{window} year moving average');
ax.plot(x, h_x, c='k', label='Model: Predicted trendline')

ax.set_title('Mean Air Temperature Measurements')
ax.set_xlabel('Year')
ax.set_ylabel('Temperature (degC)')
ax.set_xticks(range(df_group_year.index.min(), df_group_year.index.max(), 10))
ax.legend();