import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression

df = pd.read_csv('synth_temp.csv')
df = df.loc[df.Year > 1901] # Filter data with > 1901
df_group_year = df.groupby('Year').agg(np.mean) # Group by mean aggregated
window = 10
rolling = df_group_year.AverageTemperature.rolling(window).mean()

model = LinearRegression()
model.fit(df_group_year.index.values.reshape((-1, 1)), df_group_year.AverageTemperature)
m = model.coef_[0]
b = model.intercept_
print(f"Line equation is y={m:.3f}x + {b:.3f}")

# Predict and construct trend line
trend_x = np.array([
	df_group_year.index.values.min(),
	df_group_year.index.values.mean(),
	df_group_year.index.values.max()
]).reshape(-1, 1)
trend_y = model.predict(trend_x)
print(f"Predictions {trend_y}")

# Construct trend line for prediction
fig = plt.figure(figsize=(10, 7))
ax = fig.add_axes([1, 1, 1, 1])
ax.scatter(df_group_year.index, df_group_year.AverageTemperature, label='Raw Data', c='k');
ax.plot(df_group_year.index, rolling, c='k', linestyle='--', label=f'{window} year moving average');
ax.plot(trend_x, trend_y, c='k', label='Model: Predicted trendline')
ax.set_title('Mean Air Temperature Measurements')
ax.set_xlabel('Year')
ax.set_ylabel('Temperature (degC)')
ax.set_xticks(range(df_group_year.index.min(), df_group_year.index.max(), 10))
ax.legend();

# Compute R2
r2 = model.score(df_group_year.index.values.reshape((-1, 1)), df_group_year.AverageTemperature)
print(f'r2 score = {r2:0.4f}')