from sklearn.linear_model import SGDRegressor
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

# Step 1 create gradient descen model
model = SGDRegressor(
    max_iter=100000,
    learning_rate='constant',
    eta0=1e-6,
    random_state=255,
    tol=1e-6,
    penalty='none',
)


df = pd.read_csv('synth_temp.csv')
df = df.loc[df.Year > 1901] # Filter data with > 1901
df_group_year = df.groupby('Year').agg(np.mean) # Group by mean aggregated
window = 10
rolling = df_group_year.AverageTemperature.rolling(window).mean()

# Step 2 Prepare input parameters then fit the model
x = df_group_year.Year / df_group_year.Year.max()
_x = x/x.max()
y_true = df_group_year.AverageTemperature.values.ravel()
model.fit(x.values.reshape((-1, 1)), y_true)

# Step 3 Make predictions
y_pred = model.predict(x.values.reshape((-1, 1)))
r2_score(y_true, y_pred)

# Step 4 Evaluate model
r2_score(y_true, y_pred)

# Step 5 Create trend line
fig = plt.figure(figsize=(10, 7))
ax = fig.add_axes([1, 1, 1, 1]);
trend_y = model.predict(_x.reshape((-1, 1)))

# Temp measurements
ax.scatter(df_group_year.index, df_group_year.AverageTemperature, label='Raw Data', c='k');
ax.plot(df_group_year.index, rolling, c='k', linestyle='--', label=f'{window} year moving average');
ax.plot(x, trend_y, c='k', label='Model: Predicted trendline')


ax.set_title('Mean Air Temperature Measurements')
ax.set_xlabel('Year')
ax.set_ylabel('Temperature (degC)')
ax.set_xticks(range(df_group_year.index.min(), df_group_year.index.max(), 10))
ax.legend();