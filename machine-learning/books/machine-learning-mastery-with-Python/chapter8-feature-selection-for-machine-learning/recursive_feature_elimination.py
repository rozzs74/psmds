# Feature Extraction with RFE

from pandas import read_csv
from numpy import set_printoptions
from sklearn.feature_selection import RFE
from sklearn.tree import DecisionTreeClassifier

def main():
	PATH = "../pima-indians-diabetes.data.csv"
	columns = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
	df = read_csv(PATH, names=columns)
	array = df.values
	X = array[:,0:8]
	Y = array[:,8]
	# Feature extraction
	model = DecisionTreeClassifier()
	rfe = RFE(estimator=model, n_features_to_select=3)
	fit = rfe.fit(X, Y)
	for i in range(X.shape[1]):
		print('Column: %d, Selected=%s, Rank: %d' % (i, rfe.support_[i], rfe.ranking_[i]))
	print(df.head(1))
if __name__ == "__main__":
	main()
