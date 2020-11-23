import statsmodels.formula.api as sm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


df = pd.read_csv('synth_temp.csv')
model = sm.ols("Year ~ AverageTemperature", data=df)
model  = model.fit()

y_pred = model.predict()

# Plot regression against actual data
plt.figure(figsize=(12, 6))
plt.plot(df["Year"], df['AverageTemperature'], 'o')           # scatter plot showing actual data
plt.plot(df['Year'], y_pred, 'r', linewidth=2)   # regression line
plt.xlabel('Year')
plt.ylabel('Temperature')
plt.show()