import seaborn as sn

from matplotlib import pyplot

from pandas import read_csv
from pandas import concat
from pandas import DataFrame
from pandas import Series
from pandas import Index

from typing import Tuple
from typing import List
from typing import Dict
from typing import Set

from numpy import ndarray

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
# 	show_plot()

def data_profiling(df) -> bool:
	df_head: DataFrame = get_peek(df, 5)
	df_tail: DataFrame = get_tail(df, 5)
	df_dimension: Tuple = get_dimension(df)
	df_columns: Index = get_columns(df)
	get_data_information(df)
	df_data_types: Series = get_data_types(df)

	# Much better results if data is cleaned
	df_descriptive_statistics: DataFrame = get_descriptive_statistics(df)
	df_correlation: DataFrame = get_correlation(df)
	df_skew: DataFrame = get_skew(df)
	df_kurtosis: DataFrame = get_kurtosis(df)
	df_unique_value: Series = get_unique_values(df)
	df_unique_values_per_column: ndarray = get_unique_values_per_column(df, "math score")
	df_null_values: Series = show_missing_values(df)
	return True

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

def get_missing_value_percentage(df):
	return round(100 * (df.isnull().sum() / len(df)), 2)

def get_unique_values_per_column(df, name) -> ndarray:
	return df[name].unique()

def get_unique_values(df) -> Series:
	return df.nunique()

def get_skew(df) -> DataFrame:
	return df.skew()

def get_correlation(df) -> DataFrame:
	return df.corr()

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

def impute_missing_value(df) -> DataFrame:
	return df.fillna(0)

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

def show_plot() -> None:
	pyplot.show()

def main() -> None:
	# Understanding the Data
	df  = read_csv("./students.csv")
	understand_data(df)

if __name__ == "__main__":
	main()

