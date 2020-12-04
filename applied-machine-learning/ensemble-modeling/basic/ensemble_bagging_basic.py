import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

#Step 1 Read dataset
df = pd.read_csv("./titanic.csv")

# Step 2 Data preprocessing the data is not clead so preprocess is intended
def preprocess(data):
	def fix_age(age):
		if np.isnan(age):
			return -1
		elif age < 1:
			return age*100
		else:
			return age
		
	data.loc[:, 'Sex'] = data.Sex.apply(lambda s: int(s == 'female'))
	data.loc[:, 'Age'] = data.Age.apply(fix_age)

	embarked = pd.get_dummies(data.Embarked, prefix='Emb')[['Emb_C','Emb_Q','Emb_S']]
	cols = ['Pclass','Sex','Age','SibSp','Parch','Fare']

	return pd.concat([data[cols], embarked], axis=1).values


# Step 3 Resampling method 80/20 (80=training, 20=test)
train, val = train_test_split(df, test_size=0.20, random_state=7)
# Train data
x_train = preprocess(train)
y_train = train['Survived'].values
# Test data
x_val = preprocess(val)
y_val = val['Survived'].values

# Step 4 Specify Hyperparameters
dt_params = {
    'criterion': 'entropy',
    'random_state': 11
}
dt = DecisionTreeClassifier(**dt_params)
dt.fit(x_train, y_train)
dt_preds_train = dt.predict(x_train)
dt_preds_val = dt.predict(x_val)

print(f"Decision tree Accuracy on training data: {accuracy_score(y_true=y_train, y_pred=dt_preds_train)}")
print(f"Decision tree Accuracy on testing data: {accuracy_score(y_true=y_val, y_pred=dt_preds_val)}")

bc_params = {
    'base_estimator': dt,
    'n_estimators': 50,
    'max_samples': 0.5,
    'random_state': 11,
    'n_jobs': -1
}
bc = BaggingClassifier(**bc_params)
bc.fit(x_train, y_train)
bc_preds_train = bc.predict(x_train)
bc_preds_val = bc.predict(x_val)

print(f"Bagging Classifier Accuracy on training data: {accuracy_score(y_true=y_train, y_pred=bc_preds_train)}")
print(f"Bagging Classifier Accuracy on testing data: {accuracy_score(y_true=y_val, y_pred=bc_preds_val)}")

rf_params = {
    'n_estimators': 100,
    'criterion': 'entropy',
    'max_features': 0.5,
    'min_samples_leaf': 10,
    'random_state': 11,
    'n_jobs': -1
}
rf = RandomForestClassifier(**rf_params)
rf.fit(x_train, y_train)
rf_preds_train = rf.predict(x_train)
rf_preds_val = rf.predict(x_val)

print(f"Random Forest Accuracy on training data: {accuracy_score(y_true=y_train, y_pred=rf_preds_train)}")
print(f"Random Forest Accuracy on testing data: {accuracy_score(y_true=y_val, y_pred=rf_preds_val)}")