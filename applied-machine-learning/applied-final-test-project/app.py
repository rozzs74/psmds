import seaborn as sn
import missingno as msno

from matplotlib import pyplot

from pandas import read_csv
from pandas import concat
from pandas.plotting import scatter_matrix
from pandas import DataFrame
from pandas import Series
from pandas import Index
from pandas import set_option

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
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import RFE

from numpy import ndarray
from numpy import arange
from numpy import set_printoptions

from typing import Tuple
from typing import List
from typing import Dict
from typing import Set

def main() -> None:
	df: DataFrame = read_csv(PATH, header=None)
	# understand_data(df)

	# Prepare Data
	n_array: ndarray = df.values	
	X: ndarray = df.iloc[:, 0 : 60].astype(float)
	Y: ndarray = df.iloc[:, 60]
	# feature_selection(X, Y)
	# baseline_models: List[set] = [
	# 	("LR", LogisticRegression()),
	# 	("LDA", LinearDiscriminantAnalysis()),
	# 	("KNN", KNeighborsClassifier()),
	# 	("CART", DecisionTreeClassifier()),
	# 	("NB", GaussianNB()),
	# 	("SVM", SVC())
	# ]
	X_train, X_validation, Y_train, Y_validation = get_resampling_data(X, Y, 10, 7)
	# baseline_results: List[set] = evaluate_algorithms_baseline(10, 7, "accuracy", X_train, Y_train, baseline_models)
	# show_whisker_plots_for_evaluation(baseline_results[0], baseline_results[1], "Evaluation Algorithms Baseline")
	pipelines: List[set]  = [
		("ScaledLR", Pipeline([("Scaler", StandardScaler()),("LR", LogisticRegression())])),
		("ScaledLDA", Pipeline([("Scaler", StandardScaler()),("LDA", LinearDiscriminantAnalysis())])),
		("ScaledKNN", Pipeline([("Scaler", StandardScaler()),("KNN", KNeighborsClassifier())])),
		("ScaledCART", Pipeline([("Scaler", StandardScaler()),("CART", LogisticRegression())])),
		("ScaledNB", Pipeline([("Scaler", StandardScaler()),("NB", GaussianNB())])),
		("ScaledSVM", Pipeline([("Scaler", StandardScaler()),("SVM", SVC())]))
	]

	scaled_results: list = evaluate_algorithms_standardize(10, 7, "accuracy", X_train, Y_train, pipelines)
	show_whisker_plots_for_evaluation(scaled_results[0], scaled_results[1], "Evaluation Algorithms Standardize")

def understand_data(df) -> None:
    is_there_any_duplicates: bool = find_duplicates(df)
    if is_there_any_duplicates:
        df_duplicates = get_duplicates_row(df, "")
        df = drop_duplicates(df)
    is_there_any_missing_values: bool = check_missing_values(df)

    if is_there_any_missing_values == True:
        df = clean_data(df)
    
    is_done: bool = data_profiling(df)
    if is_done:
        visualization(df)

def data_profiling(df) -> bool:
    df_head: DataFrame = get_peek(df, 5)
    df_tail: DataFrame = get_tail(df, 5)
    df_dimension: Tuple = get_dimension(df)
    df_columns: Index = get_columns(df)
    get_data_information(df)
    df_data_types: Series = get_data_types(df)

    df_descriptive_statistics: DataFrame = get_descriptive_statistics(df)
    df_correlation: DataFrame = get_correlation(df, "pearson")
    df_skew: DataFrame = get_skew(df)
    df_kurtosis: DataFrame = get_kurtosis(df)
    df_null_values: Series = show_missing_values(df)
    return True

def visualization(df) -> None:
	# Univariate 
	show_histogram(df)
	show_density_plots(df)

	# Multivariate 
	df_correlation: DataFrame = get_correlation(df, "pearson")
	show_correlation_plot(df_correlation)

	# Class Distribution
	df_class_distribution: DataFrame = get_class_distribution(df, 60)
	rocks: int = df_class_distribution.iloc[1]
	sonar: int = df_class_distribution.iloc[0]
	pyplot.title("Class Distribution")
	pyplot.pie([rocks, sonar], labels=["Rock", "Sonar"])
	show_plot()

def feature_selection(X, Y) -> None:
	univariate_selection(X, Y)
	recursive_feature_elmination(X, Y)
	feature_importance(X, Y)
	
def univariate_selection(X, Y) -> bool:
	# Univariate Selection
	test = SelectKBest(score_func=chi2, k=4)
	fit = test.fit(X, Y)
	print(fit.scores_)
	cols = test.get_support()
	features_df_new = X.iloc[:,cols]
	print(features_df_new)
	set_printoptions(precision=3)
	features = fit.transform(X)
	print(features[:5])
	return True

def recursive_feature_elmination(X, Y) -> bool:
	rfe = RFE(estimator=DecisionTreeClassifier(), n_features_to_select=3)
	fit = rfe.fit(X, Y)
	for i in range(X.shape[1]):
		print('Column: %d, Selected=%s, Rank: %d' % (i, rfe.support_[i], rfe.ranking_[i]))
	return True

def feature_importance(X, Y) -> bool:
	model = ExtraTreesClassifier()
	model.fit(X, Y)
	set_printoptions(precision=3)
	print(model.feature_importances_)
	return True

def clean_data(df, default=True) -> DataFrame:
    if default:
        return impute_missing_value(df)

def get_kurtosis(df) -> DataFrame:
    return df.kurtosis()

def get_resampling_data(X, Y, size, state):
	return train_test_split(X, Y, test_size=size, random_state=state)

def get_missing_value(df):
	mask = df.isnull()
	total = mask.sum()
	percent = 100 * mask.mean()
	missing_value = concat([total, percent], axis=1, join="outer", keys=["count_missing", "percentage_missing"])
	missing_value.sort_values(by="percentage_missing", ascending=False, inplace=True)
	return missing_value

def get_class_distribution(df, name):
	return df.groupby(name).size()

def get_missing_value_percentage(df):
	return round(100 * (df.isnull().sum() / len(df)), 2)

def get_unique_values_per_column(df, name) -> ndarray:
	return df[name].unique()

def get_unique_values(df) -> Series:
	return df.nunique()

def get_skew(df) -> DataFrame:
	return df.skew()

def get_correlation(df, method) -> DataFrame:
	return df.corr(method=method)

def get_descriptive_statistics(df) -> DataFrame:
	return df.describe()

def get_peek(df, n) -> DataFrame:
	return df.head(n)

def get_tail(df, n) -> DataFrame:
	return df.tail(n)

def get_dimension(df) -> Tuple:
	return df.shape

def get_data_types(df) -> Series:
	return df.dtypes

def get_data_information(df):
	return df.info()

def get_columns(df) -> Index:
	return df.columns

def get_duplicates_row(df, name, all_rows=True):
	if all_rows:
		return df[df.duplicated()]
	else:
		return df[df.duplicated(name)]

def find_duplicates(df) -> bool:
	return df.duplicated().any()

def check_missing_values(df) -> bool:
	return df.isnull().values.any()

def drop_duplicates(df):
	return df.drop_duplicates()

def show_missing_values(df) -> Series:
	return df.isnull().sum()

def run_option() -> None:
	set_option("display.width", 100)
	set_option("precision", 3)

def impute_missing_value(df) -> DataFrame:
	return df.fillna(0)

def missing_plot(df) -> None:
	mask = df.isnull()
	nullable_columns = df.columns[mask.any()].tolist()
	msno.matrix(df[nullable_columns].sample(500))
	show_plot()

def missing_heat_map(df) -> None:
	mask = df.isnull()
	nullable_columns = df.columns[mask.any()].tolist()
	msno.heatmap(df[nullable_columns], figsize=(18, 18))
	show_plot()

def show_density_plots(df) -> None:
	df.plot(kind="density", subplots=True, layout=(8, 8), sharex=False)
	show_plot()

def show_histogram(df) -> None:
	df.hist()
	show_plot()

def show_scatter_plot(df) -> None:
	scatter_matrix(df)
	show_plot()

def show_whisker_plots_for_evaluation(results, names, title) -> None:
	fig = pyplot.figure()
	fig.suptitle(title)
	ax = fig.add_subplot(111)
	pyplot.boxplot(results)
	ax.set_xticklabels(names)
	show_plot()

def show_whisker_plots(df) -> None:
	df.plot(kind="box", subplots=True, layout=(3, 3), sharex=False, sharey=False)
	show_plot()

def show_correlation_plot(correlations) -> None:
	fig = pyplot.figure()
	ax = fig.add_subplot(111)
	cax = ax.matshow(correlations, vmin=-1, vmax=1, interpolation="none")
	fig.colorbar(cax)
	show_plot()

def show_whisker_plots_for_evaluation(results, names, title) -> None:
	fig = pyplot.figure()
	fig.suptitle(title)
	ax = fig.add_subplot(111)
	pyplot.boxplot(results)
	ax.set_xticklabels(names)
	show_plot()

def show_plot() -> None:
	pyplot.show()

def evaluate_algorithms_baseline(fold, seed, metric, X, Y, models) -> list:
	evaluation_results: list = []
	model_name: list = []
	stash_models: list = []
	i: int= 0
	model_length: int = len(models)
	while i <= model_length:
		el = models[i]
		kfold = KFold(n_splits=fold, random_state=seed, shuffle=True)
		score = cross_val_score(el[1], X, Y, cv=kfold, scoring=metric)
		evaluation_results.append(score)
		model_name.append(el[0])
		stash_models.append(el[1])
		print(f"{el[0]} Mean estimated Accuracy: {score.mean()*100:.3f}%")
		print(f"{el[0]} Estimated Standard Deviation: {score.std()*100:.3f}%")
		i += 1
		if i == model_length:
			return evaluation_results, model_name, stash_models

def evaluate_algorithms_standardize(fold, seed, metric, X, Y, pipelines) -> list:
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

if __name__ == "__main__":
    PATH: str = "./dataset.csv"
    main()