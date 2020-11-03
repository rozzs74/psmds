# Feature Importance with Extra Trees Classifier

from pandas import read_csv
from sklearn.ensemble import ExtraTreesClassifier
from numpy import set_printoptions

def main():
	PATH = "../pima-indians-diabetes.data.csv"
	columns = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
	df = read_csv(PATH, names=columns)
	array = df.values
	X = array[:,0:8]
	Y = array[:,8]	
	# feature extraction
	model = ExtraTreesClassifier()
	model.fit(X, Y)
	set_printoptions(precision=3)
	print(model.feature_importances_)
	print(df.head(1))
if __name__ == "__main__":
	main()