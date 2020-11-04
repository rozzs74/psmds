# Compare Algorithms 

from pandas import read_csv
from matplotlib import pyplot
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR


def main():
	PATH = "../housing.csv"	
	columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
	df = read_csv(PATH, names=columns)
	array = df.values
	X = array[:,0:13]
	Y = array[:,13]
	models = []
	models.append(("LR", LinearRegression()))
	models.append(("RR", Ridge()))
	models.append(("LAS", Lasso()))
	models.append(("ER", ElasticNet()))
	models.append(("CART", DecisionTreeRegressor()))
	models.append(("KNN", KNeighborsRegressor()))
	models.append(("SVR", SVR()))
	return models, X, Y

if __name__ == "__main__":
	models, X, Y = main()
	results = []
	names = []
	i = 0
	while i < len(models):
		el = models[i]
		names.append(el[0])
		kfold = KFold(n_splits=10, random_state=7, shuffle=True)
		cv_results = cross_val_score(el[1], X, Y, cv=kfold, scoring="r2")
		results.append(cv_results)
		i += 1
		if i == len(models):
			fig = pyplot.figure()
			fig.suptitle("Algorithms Comparison For Regression")
			ax = fig.add_subplot(111)
			pyplot.boxplot(results)
			ax.set_xticklabels(names)
			pyplot.show()
			break