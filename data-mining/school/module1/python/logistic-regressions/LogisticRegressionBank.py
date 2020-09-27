#Logistic Regression

#Importing pandas library to read CSV data file
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

#Reading CSV data file in Python
dataset = pd.read_csv('./bank_data.csv')

# #Dividing dataset into X and y
X = dataset.iloc[:, [0,2,4,5]].values #  AGE  INCOME  HOUSEHOLD_N  CREDIT_LINES_N
y = dataset.iloc[:, -1].values # DEAFULTED

#Importing train_test_split from sklearn.model_selection to split data into training and testing sets  # resampling 80/20
X_train, X_test, y_train,  y_test = train_test_split(X,y,test_size=0.2,random_state=0)


# #Importing StandardScaler from sklearn.preprocessing to scale matrix of features
# scale data or STD
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# #Importing LogisticRegression from sklearn.linear_model to build LogisticRegression classifier
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)
# print(y_pred)

cm = confusion_matrix(y_test, y_pred)
# print(cm)