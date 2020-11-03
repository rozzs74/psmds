# Feature Extraction with Univariate Statistical Tests (Chi-squared for classification)


from pandas import read_csv
from numpy import set_printoptions
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2


def main():
	PATH = "../pima-indians-diabetes.data.csv"
	columns = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
	df = read_csv(PATH, names=columns)
	array = df.values
	X = array[:,0:8]
	Y = array[:,8]

	# Feature selection by getting 4 attributes
	test = SelectKBest(score_func=chi2, k=4)
	fit = test.fit(X, Y)

	# Summarize the scores
	set_printoptions(precision=3)
	print(fit.scores_)
	features = fit.transform(X)
	# Summarize the selected features
	print(features[0: 5, :]) #plas, test, mass, age

if __name__ == "__main__":
	main()