import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

# Step 1 Read the data
df = pd.read_csv('linear_classifier.csv')
df.head()

# Step 2 Plot the classes using scatter plots
plt.figure(figsize=(10, 7))
for label, label_class in df.groupby('labels'):
    plt.scatter(label_class.values[:,0], label_class.values[:,1],label=f'Class {label}', marker=label, c='k')
plt.legend()
plt.title("Linear Classifier");
plt.show()

# Step 3 Fit the model

model = LinearRegression()
model.fit(df.x.values.reshape((-1, 1)), df.y.values.reshape((-1, 1)))

m = model.coef_[0][0]
b = model.intercept_[0]
print(f'y = {m}x + {b}')
# Step 4 Plot the trend line
trend = model.predict(np.linspace(0, 10).reshape((-1, 1)))
plt.figure(figsize=(10, 7))
for label, label_class in df.groupby('labels'):
    plt.scatter(label_class.values[:,0], label_class.values[:,1],label=f'Class {label}', marker=label, c='k')
plt.plot(np.linspace(0, 10), trend, c='k', label='Trendline')
plt.legend()
plt.title("Linear Classifier");
plt.show()

# Step 5 make predictions
y_pred = model.predict(df.x.values.reshape((-1, 1)))
pred_labels = []

for _y, _y_pred in zip(df.y, y_pred):
    if _y < _y_pred:
        pred_labels.append('o')
    else:
        pred_labels.append('x')
df['Pred Labels'] = pred_labels
df.head()

# Step 6 Plot predictions with trend line
plt.figure(figsize=(10, 7))
for idx, label_class in df.iterrows():
    if label_class.labels != label_class['Pred Labels']:
        label = 'D'
        s=70
    else:
        label = label_class.labels
        s=50
    plt.scatter(label_class.values[0], label_class.values[1],label=f'Class {label}', marker=label, c='k', s=s)
    
plt.plot(np.linspace(0, 10), trend, c='k', label='Trendline')
plt.title("Linear Classifier");
incorrect_class = mlines.Line2D([], [], color='k', marker='D',markersize=10, label='Incorrect Classification');
plt.legend(handles=[incorrect_class]);
plt.show()