#Logistic Regression

#Importing pandas library to read CSV data file
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

# Step 1 Load the dataset
dataset = pd.read_csv('./bank_data.csv')

# Step 2 Feature selection
X = dataset.iloc[:, [0,2,4,5]].values #  AGE  INCOME  HOUSEHOLD_N  CREDIT_LINES_N
y = dataset.iloc[:, -1].values # DEAFULTED

# Step 3 Split dataset into training and test dataset with 80/20 rule resampling method
X_train, X_test, y_train,  y_test = train_test_split(X,y,test_size=0.2,random_state=0)

# Step 4 Prepare the data
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# Step 5 Fit and train a model
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)

# Step 6 Make prediction
y_pred = classifier.predict(X_test)

# Step 7 Measure the performance of the model
cm = confusion_matrix(y_test, y_pred)