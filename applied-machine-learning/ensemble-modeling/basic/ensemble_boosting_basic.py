import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

#Step 1 Read dataset
df = pd.read_csv("../titanic.csv")

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


# Ada Boost
# dt_params = {
#     'max_depth': 1,
#     'random_state': 11
# }
# dt = DecisionTreeClassifier(**dt_params)

# ab_params = {
#     'n_estimators': 100,
#     'base_estimator': dt,
#     'random_state': 11
# }
# ab = AdaBoostClassifier(**ab_params)

# ab.fit(x_train, y_train)
# ab_preds_train = ab.predict(x_train)
# ab_preds_val = ab.predict(x_val)

# print(f"Ada Boost Accuracy on training data: {accuracy_score(y_true=y_train, y_pred=ab_preds_train)}")
# print(f"Ada Boost Accuracy on testing data: {accuracy_score(y_true=y_val, y_pred=ab_preds_val)}")

# ab_params = {
#     'base_estimator': dt,
#     'random_state': 11
# }

# n_estimator_values = list(range(10, 210, 10))
# train_accuracies, val_accuracies = [], []

# for n_estimators in n_estimator_values:
#     ab = AdaBoostClassifier(n_estimators=n_estimators, **ab_params)
#     ab.fit(x_train, y_train)
#     ab_preds_train = ab.predict(x_train)
#     ab_preds_val = ab.predict(x_val)
    
#     train_accuracies.append(accuracy_score(y_true=y_train, y_pred=ab_preds_train))
#     val_accuracies.append(accuracy_score(y_true=y_val, y_pred=ab_preds_val))

# plt.figure(figsize=(10,7))
# plt.plot(n_estimator_values, train_accuracies, label='Train')
# plt.plot(n_estimator_values, val_accuracies, label='Validation')
# plt.ylabel('Accuracy score')
# plt.xlabel('n_estimators')
# plt.legend()
# plt.show()


# Gradient Boosting
gbc_params = {
    'n_estimators': 100,
    'max_depth': 3,
    'min_samples_leaf': 5,
    'random_state': 11
}
gbc = GradientBoostingClassifier(**gbc_params)
gbc.fit(x_train, y_train)
gbc_preds_train = gbc.predict(x_train)
gbc_preds_val = gbc.predict(x_val)

print(f"Gradient Boosting Classifier Accuracy on training data: {accuracy_score(y_true=y_train, y_pred=gbc_preds_train)}")
print(f"Gradient Boosting Classifier Accuracy on testing data: {accuracy_score(y_true=y_val, y_pred=gbc_preds_val)}")