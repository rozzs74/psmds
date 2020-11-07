import numpy
import missingno as msno

from matplotlib import pyplot
from pandas import read_csv
from pandas.plotting import scatter_matrix
from pandas import set_option
from pandas import concat

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

def main():
	PATH = "./pima-indians-diabetes.data.csv"
	columns = get_columns()
	df = read_csv(PATH, names=columns)
	# Data profiling
	df_head = get_peek(df, 5)
	df_dimension = get_dimension(df)
	df_data_types = get_data_types(df)
	df_info = get_data_information(df)
	df_descriptive_statistics = get_descriptive_statistics(df)
	df_missing_value = get_missing_value(df)
	missing_plot(df)
	missing_heat_map(df)
	df = impute_missing_value(df)
	df_head = get_peek(df, 5)
	get_descriptive_statistics(df)

	# Univariate Plots
	show_histogram(df)
	show_density_plots(df)
	show_whisker_plots(df)

	# Multivariate plots
	df_correlation = get_correlation(df)
	show_correlation_plot(df_correlation, columns)
	df_skew = get_skew(df)
	df_kurtosis = get_kurtosis(df)
	show_scatter_plot(df)

	# Target Variables
	df_class_distribution = get_class_distribution(df, "diabetes")

	# Class distribution
	df["diabetes"].value_counts().plot(kind="bar")
	pyplot.title("Targe Variable (0=Negative Diabets, 1= Positive Diabetes)")
	show_plot()

	positive = df_class_distribution.iloc[1]
	negative = df_class_distribution.iloc[0]
	pyplot.title("Diabetes class distribution")
	pyplot.pie([positive, negative], labels=["Positive", "Negative"])
	show_plot()

	n_array = df.values
	X = n_array[:, 0: 8]
	Y = n_array[:, 8]
	X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=0.20, random_state=7)
	models = [("KNN", KNeighborsClassifier()), ("CART", DecisionTreeClassifier())]
	model_results, model_name, stash_models = evaluate_algorithms(10, 7, "accuracy", X_train, Y_train, models)
	show_whisker_plots_for_evaluation(model_results, model_name, "KNN vs. CART")

	# KNN 
	knn_fit = stash_models[0]
	knn_fit.fit(X_train, Y_train)
	knn_predictions = knn_fit.predict(X_validation)
	print(f"KNN Prediction Accuracy: {accuracy_score(Y_validation, knn_predictions) * 100:.2f}%")

	cart_fit = stash_models[1]
	cart_fit.fit(X_train, Y_train)
	cart_predictions = cart_fit.predict(X_validation)
	print(f"CART Prediction Accuracy: {accuracy_score(Y_validation, cart_predictions) * 100:.2f}%")

	knn_confusion_matrix = confusion_matrix(Y_validation, knn_predictions)
	print(f"KNN Confusion Matrix: {knn_confusion_matrix}")

	cart_confusion_matrix = confusion_matrix(Y_validation, cart_predictions)
	print(f"CART Confusion Matrix: {cart_confusion_matrix}")

	knn_classification_report = classification_report(Y_validation, knn_predictions)
	print(f"KNN Classification Report: {knn_classification_report}")

	cart_classification_report = classification_report(Y_validation, cart_predictions)
	print(f"CART Classification Report: {cart_classification_report}")

def evaluate_algorithms(fold, seed, metric, X, Y, models):
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
		print(f"Mean estimated Accuracy: {score.mean()*100:.3f}%")
		print(f"Estimated Standard Deviation: {score.std()*100:.3f}%")
		i += 1
		if i == len(models):
			return evaluation_results, model_name, stash_models

def get_columns():
	return ["pregnant", "glucose", "pressure", "triceps", "insulin", "mass", "pedigree", "age", "diabetes"]

def get_peek(df, n):
	return df.head(n)

def get_dimension(df):
	return df.shape

def get_data_types(df):
	return df.dtypes

def get_data_information(df):
	return df.info()

def get_descriptive_statistics(df, with_option=True):
	if with_option:
		run_option()
	return df.describe()

def get_correlation(df):
	return df.corr()

def get_skew(df):
	return df.skew()

def get_kurtosis(df):
    return df.kurtosis()

def get_class_distribution(df, name):
	return df.groupby(name).size()

def get_missing_value(df):
	mask = df.isnull()
	total = mask.sum()
	percent = 100 * mask.mean()
	missing_value = concat([total, percent], axis=1, join="outer", keys=["count_missing", "percentage_missing"])
	missing_value.sort_values(by="percentage_missing", ascending=False, inplace=True)
	return missing_value

def run_option():
	set_option("display.width", 100)
	set_option("precision", 3)

def impute_missing_value(df):
	return df.fillna(0)

def missing_plot(df):
	mask = df.isnull()
	nullable_columns = df.columns[mask.any()].tolist()
	msno.matrix(df[nullable_columns].sample(500))
	show_plot()

def missing_heat_map(df):
	mask = df.isnull()
	nullable_columns = df.columns[mask.any()].tolist()
	msno.heatmap(df[nullable_columns], figsize=(18, 18))
	show_plot()
def show_density_plots(df):
	df.plot(kind="density", subplots=True, layout=(3, 3), sharex=False)
	show_plot()

def show_histogram(df):
	df.hist()
	show_plot()

def show_scatter_plot(df):
	scatter_matrix(df)
	show_plot()

def show_whisker_plots_for_evaluation(results, names, title):
	fig = pyplot.figure()
	fig.suptitle(title)
	ax = fig.add_subplot(111)
	pyplot.boxplot(results)
	ax.set_xticklabels(names)
	show_plot()

def show_whisker_plots(df):
	df.plot(kind="box", subplots=True, layout=(3, 3), sharex=False, sharey=False)
	show_plot()

def show_correlation_plot(correlations, names):
	fig = pyplot.figure()
	ax = fig.add_subplot(111)
	cax = ax.matshow(correlations, vmin=-1, vmax=1)
	fig.colorbar(cax)
	ticks = numpy.arange(0, 9, 1)
	ax.set_xticks(ticks)
	ax.set_yticks(ticks)
	ax.set_xticklabels(names) 
	ax.set_yticklabels(names)
	show_plot()

def show_plot():
	pyplot.show()

if __name__ == "__main__":
	main()