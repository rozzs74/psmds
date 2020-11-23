import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression

df = pd.read_csv('synth_temp.csv')

# Filter data with greater than 1901
df = df.loc[df.Year > 1901]

# Moving average
window = 10
rolling = df_group_year.AverageTemperature.rolling(window).mean()
df_group_year['Year'] = df_group_year.index
model = LinearRegression()

# Squared each values
df_group_year['Year'] = df_group_year.index
df_group_year['Year2'] = df_group_year.index ** 2
# Joined them
df_group_year[['Year', 'Year2']]

model.fit(df_group_year[['Year2', 'Year']], df_group_year.AverageTemperature)
a = model.coef_[0]
m = model.coef_[1]
c = model.intercept_
print(f"Model definition y={a:0.4f}x^2 + {m:0.4f}x + {c:0.4f}")
# Model evaluation
r2 = model.score(df_group_year[['Year2', 'Year']], df_group_year.AverageTemperature)
print(f'r2 score = {r2:0.4f}')

# Use linspace to get a range of values, in 20 yr increments
x = np.linspace(df_group_year['Year'].min(), df_group_year['Year'].max(), 20)

# Gen 20 x 2
trend_x = np.zeros((20, 2))
trend_x[:,0] = x ** 2 # Assign to the first column
trend_x[:,1] = x # Assign to the second column

trend_y = model.predict(trend_x)
print(trend_y)

# Construct trend line of the prediction
fig = plt.figure(figsize=(10, 7))
ax = fig.add_axes([1, 1, 1, 1]);
ax.scatter(df_group_year.index, df_group_year.AverageTemperature, label='Raw Data', c='k');
ax.plot(df_group_year.index, rolling, c='k', linestyle='--', label=f'{window} year moving average');
ax.plot(trend_x[:,1], trend_y, c='k', label='Model: Predicted trendline')

ax.set_title('Mean Air Temperature Measurements')
ax.set_xlabel('Year')
ax.set_ylabel('Temperature (degC)')
ax.set_xticks(range(df_group_year.index.min(), df_group_year.index.max(), 10))
ax.legend();