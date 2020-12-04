import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

from math import sqrt


house_prices_reg = pd.read_csv("./houseprices_regression.csv")
hp_head = house_prices_reg.head()
titanic_clf = pd.read_csv("./titanic_classification.csv")
tnc_head = titanic_clf.head()


#Regression metrics
# reg = pickle.load(open("./saved_models/stacked_linear_regression.pkl", "rb"))
# X = house_prices_reg.drop(columns=["y"])
# y = house_prices_reg["y"].values
# y_pred = reg.predict(X)
# print('Mean Absolute Error = {}'.format(mean_absolute_error(y, y_pred)))
# print('Root Mean Squared Error = {}'.format(sqrt(mean_squared_error(y, y_pred))))
# print('R Squared Score = {}'.format(r2_score(y, y_pred)))


# Classification Metrics
clf = pickle.load(open("./saved_models/random_forest_clf.pkl", "rb"))
X = titanic_clf.iloc[:, :-1].values
y = titanic_clf.iloc[:, -1].values
y_pred = clf.predict(X)
y_pred_probs = clf.predict_proba(X)[:, 1]
print(f"Accuracy score {accuracy_score(y, y_pred)}")
print(confusion_matrix(y_pred=y_pred, y_true=y))

print('Precision Score = {}'.format(precision_score(y, y_pred)))
print('Recall Score = {}'.format(recall_score(y, y_pred)))
print('F1 Score = {}'.format(f1_score(y, y_pred)))