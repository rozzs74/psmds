# Compare Algorithms
from pandas import read_csv
from matplotlib import pyplot
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

def main():
	PATH = "../pima-indians-diabetes.data.csv"
	columns = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
	df = read_csv(PATH, names=columns)
	array = df.values
	X = array[:,0:8]
	Y = array[:,8]	
	models = []
	models.append(("LR", LogisticRegression(solver="lbfgs", max_iter=400)))
	models.append(("LDA", LinearDiscriminantAnalysis()))
	models.append(("KNN", KNeighborsClassifier()))
	models.append(("CART", DecisionTreeClassifier()))
	models.append(("NB", GaussianNB()))
	models.append(("SVM", SVC()))
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
		cv_results = cross_val_score(el[1], X, Y, cv=kfold, scoring="accuracy")
		results.append(cv_results)
		i += 1
		if i == len(models):
			fig = pyplot.figure()
			fig.suptitle("Algorithms Comparison For Classification")
			ax = fig.add_subplot(111)
			pyplot.boxplot(results)
			ax.set_xticklabels(names)
			pyplot.show()
			break
