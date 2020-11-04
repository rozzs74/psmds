
# Bagged Decisition Trees for Classification

from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import ExtraTreesClassifier

def main():
	PATH = "../pima-indians-diabetes.data.csv"
	columns = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
	df = read_csv(PATH, names=columns)
	array = df.values
	X = array[:,0:8]
	Y = array[:,8]
	kfold = KFold(n_splits=10, random_state=7, shuffle=True)
	model = ExtraTreesClassifier(n_estimators=100, max_features=7)
	results = cross_val_score(model, X, Y, cv=kfold, scoring="accuracy")
	print(round(results.mean()*100, 3))

if __name__ == "__main__":
	main()