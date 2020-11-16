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
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier

from numpy import ndarray
from numpy import arange

from typing import Tuple
from typing import List
from typing import Dict
from typing import Set

def main() -> None:
    df: DataFrame = read_csv(PATH, header=None)
    understand_data(df)

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
	# Missingness
	missing_heat_map(df)
	missing_plot(df)

	# Univariate 
	show_histogram(df)
	show_density_plots(df)

	# Multivariate 
	df_correlation = get_correlation(df, "pearson")
	show_correlation_plot(df_correlation)

def clean_data(df, default=True) -> DataFrame:
    if default:
        return impute_missing_value(df)

def get_kurtosis(df) -> DataFrame:
    return df.kurtosis()

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

def show_plot() -> None:
	pyplot.show()

if __name__ == "__main__":
    PATH: str = "./sonar.csv"
    main()