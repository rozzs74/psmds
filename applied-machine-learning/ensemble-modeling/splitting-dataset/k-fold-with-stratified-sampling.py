import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

#Step 1 Read dataset
df = pd.read_csv("./titanic.csv")


# Step 2 Prepare the dataset


X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values
skf = StratifiedKFold(n_splits=5)

scores = []

for train_index, val_index in skf.split(X, y):
    X_train, X_val = X[train_index], X[val_index]
    y_train, y_val = y[train_index], y[val_index]
    
    rf_skf = RandomForestClassifier(**rf.get_params())
    
    rf_skf.fit(X_train, y_train)
    y_pred = rf_skf.predict(X_val)
    
    scores.append(accuracy_score(y_val, y_pred))
  
print(scores)
print('Mean Accuracy Score = {}'.format(np.mean(scores)))