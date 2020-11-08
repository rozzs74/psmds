# Load libraries
import numpy

from matplotlib import pyplot
from pandas import read_csv
from pandas import set_option
from pandas import concat
from pandas.plotting import scatter_matrix
from sklearn import tree
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis 
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier


def main():
	PATH = "../sonar.csv"
	# Analyze Data (Data Profiling, Imputation strategies, and Visualization)
	df = read_csv(PATH, header=None)
	# run_data_profiling(df)
	# run_visualization(df)
	n_array = df.values
	X = n_array[:, 0 : 60].astype(float)
	Y = n_array[:, 60]
	X_train, X_validation, Y_train, Y_validation = get_resampling_data(X, Y, 10, 7)
	# Evaluate Algorithm baseline (Spot-check)
	# models = [
	# 	("LR", LogisticRegression()),
	# 	("LDA", LinearDiscriminantAnalysis()),
	# 	("KNN", KNeighborsClassifier()),
	# 	("CART", DecisionTreeClassifier()),
	# 	("NB", GaussianNB()),
	# 	("SVM", SVC())
	# ]
	# model_results, model_name, stash_models = evaluate_algorithms_baseline(10, 7, "accuracy", X_train, Y_train, models)
	# show_whisker_plots_for_evaluation(model_results, model_name, "Algorithms Comparison")


	# Evaluate Algorithms Standardize
	# pipelines = [
	# 	("ScaledLR", Pipeline([("Scaler", StandardScaler()),("LR", LogisticRegression())])),
	# 	("ScaledLDA", Pipeline([("Scaler", StandardScaler()),("LDA", LinearDiscriminantAnalysis())])),
	# 	("ScaledKNN", Pipeline([("Scaler", StandardScaler()),("KNN", KNeighborsClassifier())])),
	# 	("ScaledCART", Pipeline([("Scaler", StandardScaler()),("CART", LogisticRegression())])),
	# 	("ScaledNB", Pipeline([("Scaler", StandardScaler()),("NB", GaussianNB())])),
	# 	("ScaledSVM", Pipeline([("Scaler", StandardScaler()),("SVM", SVC())]))
	# ]

	# scaled_model_results, scaled_model_name, scaled_stash_models = evaluate_algorithms_standardize(10, 7, "accuracy", X_train, Y_train, pipelines)
	# show_whisker_plots_for_evaluation(scaled_model_results, scaled_model_name, "Scaled Algorithms Comparison")

	# Tuning scaled Algorithm (SVM, CART, LR)
	
	# SVM
	# tune_svm(X_train, Y_train)

	# CART	
	# tune_cart(X_train, Y_train)

	#LR
	# tune_logistic_regression(X_train, Y_train)

	# Ensemble
	ensembles = [
		("AB", AdaBoostClassifier()),
		("GBM", GradientBoostingClassifier()),
		("RF", RandomForestClassifier()),
		("ET", ExtraTreesClassifier())
	]
	make_ensemble_methods(ensembles, X_train, Y_train, "Ensemble Algorithm Comparison")

def make_ensemble_methods(ensembles, X, Y, title):
	ensembles_results = []
	ensembles_names = []
	j = 0

	while j <= len(ensembles):
		el = ensembles[j]
		kfold = KFold(n_splits=10, random_state=7, shuffle=True)
		cv_result = cross_val_score(el[1], X, Y, cv=kfold, scoring="accuracy")
		ensembles_results.append(cv_result)
		ensembles_names.append(el[0])
		print(f"Ensemble method:{el[0]} mean:{cv_result.mean()*100:.3f}% std:{cv_result.std()*100:.3f}%")
		j += 1
		if j == len(ensembles):
			show_whisker_plots_for_evaluation(ensembles_results, ensembles_names, title)
			break

def tune_logistic_regression(X, Y, skip_summary=False):
	pipe = Pipeline([("Scaler", StandardScaler()), ("LR", LogisticRegression())])
	param_grid = {
		"LR__C": [1.0, 0.1, 0.01, 100, 20, 10],
		"LR__solver": ["lbfgs", "newton-cg", "liblinear"],
		"LR__penalty": ["l2"],
		"LR__max_iter":[400, 500, 1000]
	}
	kfold = KFold(n_splits=10, random_state=7, shuffle=True)
	grid = GridSearchCV(estimator=pipe, param_grid=param_grid, scoring="accuracy", cv=kfold)
	grid_result = grid.fit(X, Y)
	print(f"LR best score:{grid_result.best_score_*100:.3f}% params:{grid_result.best_params_}")

	if skip_summary:
		pass
	else:
		lr_means = grid_result.cv_results_["mean_test_score"]
		lr_stds = grid_result.cv_results_["std_test_score"]
		lr_params = grid.cv_results_["params"]
		for mean, std, param in zip(lr_means, lr_stds, lr_params):
			print(f"LR mean={mean}, std={std}, param={param}")

def tune_cart(X, Y, skip_summary=False):
	pipe = Pipeline([("Scaler", StandardScaler()), ("CART", ExtraTreesClassifier())])
	param_grid = {"CART__n_estimators": numpy.array([10, 50, 100, 500, 1000])}
	model_cart = ExtraTreesClassifier()
	kfold = KFold(n_splits=10, random_state=7, shuffle= True)
	grid = GridSearchCV(estimator=pipe, param_grid=param_grid, scoring="accuracy", cv=kfold)
	grid_result = grid.fit(X, Y)
	print(f"CART best score:{grid_result.best_score_*100:.3f}% params:{grid_result.best_params_}")

	if skip_summary == False:
		cart_means = grid_result.cv_results_["mean_test_score"]
		cart_stds = grid_result.cv_results_["std_test_score"]
		cart_params = grid_result.cv_results_["params"]
		
		for mean, std, param in zip(cart_means, cart_stds, cart_params):
			print(f"CART mean={mean}, std={std}, param={param}")
	else:
		pass

def tune_svm(X, Y):
	pipe = Pipeline([("Scaler", StandardScaler()), ("SVC", SVC())])
	param_grid = {"SVC__C":  [0.1, 0.3, 0.5, 0.7, 0.9, 1.0, 1.3, 1.5, 1.7, 2.0], "SVC__kernel": ["linear", "poly", "rbf", "sigmoid"]}
	kfold = KFold(n_splits=10, random_state=7, shuffle=True)
	grid = GridSearchCV(estimator=pipe, param_grid=param_grid, scoring="accuracy", cv=kfold)
	grid_result = grid.fit(X, Y)
	print(f"Best score{grid_result.best_score_} Params:{grid_result.best_params_}")
	svm_means = grid_result.cv_results_["mean_test_score"]
	svm_stds = grid_result.cv_results_["std_test_score"]
	svm_params = grid_result.cv_results_["params"]
	for mean, std, param in zip(svm_means, svm_stds, svm_params):
		print(f"SVM mean={mean}, std={std}, param={param}")

def show_whisker_plots_for_evaluation(results, names, title):
	fig = pyplot.figure()
	fig.suptitle(title)
	ax = fig.add_subplot(111)
	pyplot.boxplot(results)
	ax.set_xticklabels(names)
	show_plot()

def evaluate_algorithms_standardize(fold, seed, metric, X, Y, pipelines):
	i = 0
	model_results = []
	model_name = []
	stash_models = []
	pipelines_length = len(pipelines)
	while i <= pipelines_length:
		el = pipelines[i]
		kfold = KFold(n_splits=10, random_state=7, shuffle=True)
		cv_results = cross_val_score(el[1], X, Y, cv=kfold, scoring=metric)
		print(f"Name:{el[0]} Mean:{cv_results.mean()*100:.3f}% STD:{cv_results.std()*100:.3f}%")
		model_results.append(cv_results)
		model_name.append(el[0])
		stash_models.append(el[1])
		i += 1
		if i == pipelines_length:
			return model_results, model_name, stash_models


def evaluate_algorithms_baseline(fold, seed, metric, X, Y, models):
	evaluation_results = []
	model_name = []
	stash_models = []
	i = 0
	while i <= len(models):
		el = models[i]
		kfold = KFold(n_splits=fold, random_state=seed, shuffle=True)
		score = cross_val_score(el[1], X, Y, cv=kfold, scoring=metric)
		evaluation_results.append(score)
		model_name.append(el[0])
		stash_models.append(el[1])
		print(f"{el[0]} Mean estimated Accuracy: {score.mean()*100:.3f}%")
		print(f"{el[0]} Estimated Standard Deviation: {score.std()*100:.3f}%")
		i += 1
		if i == len(models):
			return evaluation_results, model_name, stash_models
	
def run_data_profiling(df):
	df_head = get_peek(df, 5)
	# df_tail = get_tail(df, 5)
	# df_dimension = get_dimension(df)
	# df_no_of_rows = df_dimension[0]
	# df_no_of_columns = df_dimension[1]
	# df_data_types = get_data_types(df)
	# # df_info = get_data_information(df)
	# df_descriptive_statistics = get_descriptive_statistics(df)
	# df_missing_value = get_missing_value(df)
	return True

def run_visualization(df):
	# Visualization
	# Unimodal Data Visualization
	# show_histogram(df)
	# show_density_plots(df)

	# Multimodal
	# df_correlation = get_correlation(df, "pearson")
	# show_correlation_matrix(df_correlation)
	return True

def get_resampling_data(X, Y, size, state):
	return train_test_split(X, Y, test_size=size, random_state=state)


def show_correlation_matrix(corr):
	fig = pyplot.figure()
	ax = fig.add_subplot(111)
	cax = ax.matshow(corr, vmin=-1, vmax=1, interpolation="none")
	fig.colorbar(cax)
	show_plot()

def get_peek(df, n):
	return df.head(n)

def get_tail(df, n):
	return df.tail(n)

def get_correlation(df, method):
	return df.corr(method=method)

def get_dimension(df):
	return df.shape

def get_missing_value(df):
	mask = df.isnull()
	total = mask.sum()
	percent = 100 * mask.mean()
	missing_value = concat([total, percent], axis=1, join="outer", keys=["count_missing", "percentage_missing"])
	missing_value.sort_values(by="percentage_missing", ascending=False, inplace=True)
	return missing_value

def get_data_types(df):
	return df.dtypes

def get_data_information(df):
	return df.info()

def get_descriptive_statistics(df):
	return df.describe()

def show_histogram(df):
	df.hist()
	show_plot()


def show_density_plots(df):
	df.plot(kind="density", subplots=True, layout=(8, 8), sharex=False)
	show_plot()


def show_plot():
	pyplot.show()

if __name__ == "__main__":
	main()