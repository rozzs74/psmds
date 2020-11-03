
# Cross Validation Classification Confusion Matrix

from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

def main():
    PATH = "../pima-indians-diabetes.data.csv"
    columns = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
    df = read_csv(PATH, names=columns)
    array = df.values
    X = array[:,0:8]
    Y = array[:,8]
    test_size = 0.33
    seed = 7
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)
    model = LogisticRegression(solver="lbfgs", max_iter=1000)
    model.fit(X_train, Y_train)
    predicted = model.predict(X_test)
    matrix = confusion_matrix(Y_test, predicted)
    print(matrix)


if __name__ == "__main__":
	main()