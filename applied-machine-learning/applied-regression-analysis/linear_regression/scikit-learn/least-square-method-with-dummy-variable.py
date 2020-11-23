import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression

df = pd.read_csv('synth_temp.csv')
df = df.loc[df.Year > 1901]
df_group_year = df.groupby('Year').agg(np.mean)

# Moving average
window = 10
rolling = df_group_year.AverageTemperature.rolling(window).mean();

df_group_year['Year'] = df_group_year.index

model = LinearRegression()

# Dummy Variables
df_group_year['Gt_1960'] = [0 if year < 1960 else 10 for year in df_group_year.Year]
df_group_year['Gt_1945'] = [0 if year < 1945 else 10 for year in df_group_year.Year]

# Note the year values need to be provided as an N x 1 array
model.fit(df_group_year[['Year', 'Gt_1960', 'Gt_1945']], df_group_year.AverageTemperature)
m = model.coef_[0]
b = model.intercept_
print(f"Line equation is y={m:.3f}x + {b:.3f}")
r2 = model.score(df_group_year[['Year', 'Gt_1960', 'Gt_1945']], df_group_year.AverageTemperature)
print(f'r2 score = {r2:0.4f}')

# Use linspace to get a range of values, in 20 yr increments
x = np.linspace(df_group_year['Year'].min(), df_group_year['Year'].max(), 20)
# Create array of zero in shape of 20 x 3
trend_x = np.zeros((20, 3))
trend_x[:,0] = x # Assign to the first column
trend_x[:,1] = [10 if _x > 1960 else 0 for _x in x] # Assign to the second column
trend_x[:,2] = [10 if _x > 1945 else 0 for _x in x] # Assign to the third column
trend_y = model.predict(trend_x)

fig = plt.figure(figsize=(10, 7))
ax = fig.add_axes([1, 1, 1, 1]);

# Temp measurements
ax.scatter(df_group_year.index, df_group_year.AverageTemperature, label='Raw Data', c='k');
ax.plot(df_group_year.index, rolling, c='k', linestyle='--', label=f'{window} year moving average');
ax.plot(trend_x[:,0], trend_y, c='k', label='Model: Predicted trendline')

ax.set_title('Mean Air Temperature Measurements')
ax.set_xlabel('Year')
ax.set_ylabel('Temperature (degC)')
ax.set_xticks(range(df_group_year.index.min(), df_group_year.index.max(), 10))
ax.legend();
