
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from statsmodels.tsa.ar_model import AR

# Step 1 read and manipulate dataset
df = pd.read_csv('spx.csv')
yr = []
for x in df.date:
    x = int(x[-2:])
    if x < 10:
        x = f'200{x}'
    elif x < 20:
        x = f'20{x}'
    else:
        x = f'19{x}'  
    yr.append(x)
df['Year'] = yr
df.head()

# Step 2 Pot the raw dataset with years along the x axis in multiples of ive
plt.figure(figsize=(10, 7))
plt.plot(df.close.values);
yrs = [yr for yr in df.Year.unique() if (int(yr[-2:]) % 5 == 0)]
plt.xticks(np.arange(0, len(df), len(df) // len(yrs)), yrs);
plt.title('S&P 500 Daily Closing Price');
plt.xlabel('Year');
plt.ylabel('Price ($)');

df.close[:10].values
df.close[:10].shift(3).values

# Step 3
plt.figure(figsize=(15, 7))
plt.plot(df.close.values, label='Original Dataset', c='k', linestyle='-');
plt.plot(df.close.shift(100), c='k', linestyle=':', label='Lag 100');
yrs = [yr for yr in df.Year.unique() if (int(yr[-2:]) % 5 == 0)]
plt.xticks(np.arange(0, len(df), len(df) // len(yrs)), yrs);
plt.title('S&P 500 Daily Closing Price');
plt.xlabel('Year');
plt.ylabel('Price ($)');
plt.legend();

# Step 4
plt.figure(figsize=(10, 7))
pd.plotting.autocorrelation_plot(df.close);

# Step 5

plt.figure(figsize=(10, 7))
ax = pd.plotting.autocorrelation_plot(df.close);
ax.set_ylim([-0.1, 0.1]);

# Step 6
plt.figure(figsize=(10,7))
ax = pd.plotting.lag_plot(df.close, lag=100);

#Step 7
plt.figure(figsize=(10,7))
ax = pd.plotting.lag_plot(df.close, lag=4000);

model = AR(df.close)
model_fit = model.fit()
print('Lag: %s' % model_fit.k_ar)
print('Coefficients: %s' % model_fit.params)

predictions = model_fit.predict(start=36, end=len(df) + 500)
predictions[:10].values

# Step 7 Plot the prediction with original dataset
plt.figure(figsize=(10, 7))
plt.plot(predictions, c='g', linestyle=':', label='Predictions');
plt.plot(df.close.values, label='Original Dataset');
yrs = [yr for yr in df.Year.unique() if (int(yr[-2:]) % 5 == 0)]
plt.xticks(np.arange(0, len(df), len(df) // len(yrs)), yrs);
plt.title('S&P 500 Daily Closing Price');
plt.xlabel('Year');
plt.ylabel('Price ($)');
plt.legend();

plt.figure(figsize=(10, 7))
plt.plot(predictions, c='g', linestyle=':', label='Predictions');
plt.plot(df.close.values, label='Original Dataset');
yrs = [yr for yr in df.Year.unique() if (int(yr[-2:]) % 5 == 0)]
plt.xticks(np.arange(0, len(df), len(df) // len(yrs)), yrs);
plt.title('S&P 500 Daily Closing Price');
plt.xlabel('Year');
plt.ylabel('Price ($)');
plt.xlim([2000, 2500])
plt.ylim([420, 500])
plt.legend();