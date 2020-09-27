#Step 1: Load Pandas library and the dataset using Pandas
import pandas as pd
dataset = pd.read_csv('cancer_data1.csv')
dataset.head()


#Step 2: Define the features and the target
X = pd.DataFrame(dataset.iloc[:,:-1])
y = pd.DataFrame(dataset.iloc[:,-1])

X
y

#Step 3: Split the dataset into train and test sklearn
from sklearn.model_selection import train_test_split
X_train,X_test,y_train, y_test = train_test_split(X,y,test_size=0.20)


#Step 4: Import the random forest classifier function from sklearn ensemble module. Build the random forest classifier model with the help of the random forest classifier function
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=20, criterion ='gini', random_state=1, max_depth=3)
classifier.fit(X_train, y_train.values.ravel())

#model = forest.fit(train_fold, train_y.values.ravel())

#Step 5: Predict values using the random forest classifier model
y_pred = classifier.predict(X_test)


#Step 6: Evaluate the random forest classifier model
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(accuracy_score(y_test, y_pred)) 

#Step 7: Let us find out important features and visualize them using Seaborn
import pandas as pd
feature_imp = pd.Series(classifier.feature_importances_,index=X.columns).sort_values(ascending=False)
feature_imp

import matplotlib.pyplot as plt
import seaborn as sns
#%matplotlib inline
#Creating a bard plot
sns.barplot(x=feature_imp, y=feature_imp.index)
plt.xlabel("Feature Importance Score")
plt.ylabel("Features")
plt.title("Visualizing Important Features")
plt.show()


#Step 8: Import the SelectFromModel function. We will pass the classifier object we’ve created above. Also, we will add a threshold value of 0.1
from sklearn.feature_selection import SelectFromModel
feat_sel = SelectFromModel(classifier, threshold =0.1)
feat_sel.fit(X_train, y_train.values.ravel())

#Step 9: With the help of the ‘transform’ method, we will pick the important features and store them in new train and test objects
X_imp_train = feat_sel.transform(X_train)
X_imp_test = feat_sel.transform(X_test)


#Step 10: Let us now build a new random forest classifier model (so that we can compare the results of this model with the old one)
clf_imp = RandomForestClassifier(n_estimators=20, criterion='gini', random_state=1, max_depth = 7)
#clf_imp.fit(X_imp_train, y_train)
clf_imp.fit(X_imp_train, y_train.values.ravel())


#Step 11: Let us see the accuracy result of the old model
y_pred = classifier.predict(X_test)
accuracy_score(y_test, y_pred)


#Step 12: Let us see the accuracy result of the new model after feature selection
y_imp_pred = clf_imp.predict(X_imp_test)
accuracy_score(y_test, y_imp_pred)

#https://intellipaat.com/blog/what-is-random-forest-algorithm-in-python/#Build-Random-Forest-Regression-Model






