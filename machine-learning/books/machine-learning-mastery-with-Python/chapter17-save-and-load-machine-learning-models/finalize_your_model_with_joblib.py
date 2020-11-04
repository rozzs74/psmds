# Save Model Using Joblib

from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from joblib import dump
from joblib import load

def main():
	PATH = "../pima-indians-diabetes.data.csv"
	columns = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
	df = read_csv(PATH, names=columns)
	array = df.values
	X = array[:,0:8]
	Y = array[:,8]
	X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=7)
	model = LogisticRegression(solver="lbfgs", max_iter=1000)
	model.fit(X_train, Y_train)
	filename = "finalized_model_joblib.sav"
	dump(model, filename)

	loaded_model = load(filename)
	result = loaded_model.score(X_test, Y_test)
	print(result)

if __name__ == "__main__":
	main()