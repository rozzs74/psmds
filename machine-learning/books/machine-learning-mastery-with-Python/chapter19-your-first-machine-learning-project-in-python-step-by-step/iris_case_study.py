from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

def main():
	PATH = "../iris.csv"
	columns = ["sepal-length", "sepal-width", "petal-length", "petal-width", "class"]
	df = read_csv(PATH, names=columns)
	df_dimension = df.shape
	df_head = df.head(5)
	df_summary = df.describe()
	df_class_distribution = df.groupby("class").size()
	
	# Univariate plots
	# df.plot(kind="box", subplots=True, layout=(2, 2), sharex=False, sharey=False)
	# pyplot.show()
	# df.hist()
	# pyplot.show()
	# End univariate plots

	# Multivariate Plots
	# scatter_matrix(df)
	# pyplot.show()
	
	array = df.values
	X = array[:, 0 : 4]
	Y = array[:, 4]


	# Spot-Check Algorithm
	# models = [("LR", LogisticRegression(solver="lbfgs", max_iter=1000)), ("LDA", LinearDiscriminantAnalysis()), ("KNN", KNeighborsClassifier()), ("CART", DecisionTreeClassifier()), ("NB", GaussianNB()), ("SVM", SVC())]

	# model_results = []
	# model_names = []

	# i = 0

	# while i < len(models):
	# 	el = models[i]
	# 	kfold = KFold(n_splits=10, random_state=7, shuffle=True)
	# 	cv_results = cross_val_score(el[1], X_train, Y_train, cv=kfold, scoring="accuracy")
	# 	print(f"model: {el[0]}, results: mean={cv_results.mean() * 100:.3f} std={cv_results.std()*100:.3f}")
	# 	model_results.append(cv_results)
	# 	model_names.append(el[0])
	# 	i += 1
	# 	if i == len(models):
	# 		# fig = pyplot.figure()
	# 		# fig.suptitle("Algorithm Comparison")
	# 		# ax = fig.add_subplot(111)
	# 		# pyplot.boxplot(model_results)
	# 		# ax.set_xticklabels(model_names)
	# 		# pyplot.show()
	# 		break
	X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, random_state=7)
	m_knn = KNeighborsClassifier()
	m_knn.fit(X_train, Y_train)
	predictions = m_knn.predict(X_test)
	print(f"Accuracy {accuracy_score(Y_test, predictions) * 100:.2f}%")
	print(f"Confusion Matrix {confusion_matrix(Y_test, predictions)}")
	print(f"Classification report {classification_report(Y_test, predictions)}")

if __name__ == "__main__":
	main()