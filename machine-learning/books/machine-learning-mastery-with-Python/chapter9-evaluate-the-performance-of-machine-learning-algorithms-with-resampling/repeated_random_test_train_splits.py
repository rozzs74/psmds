
# Evaluate using Shuffle Split Cross Validation

from pandas import read_csv
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression

def main():
	PATH = "../pima-indians-diabetes.data.csv"
	columns = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
	df = read_csv(PATH, names=columns)
	array = df.values
	X = array[:,0:8]
	Y = array[:,8]	
	n_splits = 10
	test_size = 0.33
	seed = 7
	kfold = ShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=seed)
	model = LogisticRegression(solver="lbfgs", max_iter=1000)
	results = cross_val_score(model, X, Y, cv=kfold)
	print(f"Accuracy mean:{results.mean()*100:.3f}%")
	print(f"Accuracy STD: {results.std()*100:.3f}%")
if __name__ == "__main__":
	main()