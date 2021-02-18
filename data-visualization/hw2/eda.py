import numpy
import missingno as msno
import seaborn as sn

from matplotlib import pyplot

from pandas import read_csv
from pandas import DataFrame
from pandas import Series
from pandas import Index
from pandas import concat
from pandas.plotting import scatter_matrix
from pandas import set_option

from typing import Tuple
from typing import List
from typing import Dict
from typing import Set

from numpy import ndarray

def get_custom_columns() -> List[str]:
    return ["pregnant", "glucose", "pressure", "triceps", "insulin", "mass", "pedigree", "age", "diabetes"]

def get_peek(df, n) -> DataFrame:
	return df.head(n)

def get_tail(df, n) -> DataFrame:
	return df.tail(n)

def get_missing_value(df) -> DataFrame:
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

def get_kurtosis(df) -> DataFrame:
    return df.kurtosis()

def get_descriptive_statistics(df) -> DataFrame:
	return df.describe()

def get_class_distribution(df, name) -> DataFrame:
	return df.groupby(name).size()

def get_dimension(df) -> Tuple:
	return df.shape

def get_data_information(df):
	return df.info()

def get_data_types(df) -> Series:
	return df.dtypes

def get_columns(df) -> Index:
	return df.columns

def get_duplicates_row(df, name, all_rows=True):
	if all_rows:
		return df[df.duplicated()]
	else:
		return df[df.duplicated(name)]

def find_duplicates(df) -> bool:
	return df.duplicated().any()

def drop_duplicates(df):
	return df.drop_duplicates()

def show_missing_values(df) -> Series:
	return df.isnull().sum()

def check_missing_values(df) -> bool:
	return df.isnull().values.any()

def clean_data(df, default=True) -> DataFrame:
	if default:
		return impute_missing_value(df)

def impute_missing_value(df) -> DataFrame:
	return df.fillna(0)

def read_missing_values(df) -> None:
    for i in range(df.shape[1]):
        n_miss = df[[i]].isnull().sum()
        perc = n_miss / df.shape[0] * 100
        print('> %d, Missing: %d (%.1f%%)' % (i, n_miss, perc))

def show_density_plots(df) -> None:
	df.plot(kind="density", subplots=True, layout=(3, 3), sharex=False)
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
def show_pair_plot(df):
	sn.pairplot(df,hue='Outcome')
def show_plot() -> None:
	pyplot.show()




def main():
	#Data Profiling
	PATH: str = "./dataset.csv"
	columns: List[str] = get_custom_columns()
	df: DataFrame = read_csv(PATH, header=None, na_values='?')
	df_head: DataFrame = get_peek(df, 10)
	read_missing_values(df)
	df_dimension: Tuple = get_dimension(df)
	pyplot.barh(["Features", "Instances"], [df_dimension[1], df_dimension[0]])
	show_plot()
	df_data_types: Series = get_data_types(df)
	df_info = get_data_information(df)
	df_descriptive_statistics: DataFrame = get_descriptive_statistics(df)
	set_option("display.width", 100)
	set_option("precision", 3)
	df_descriptive_statistics

	# Imputation
	df_missing_value: DataFrame = get_missing_value(df)
	missing_plot(df)
	missing_heat_map(df)
	df: DataFrame = impute_missing_value(df)
	columns: List[str] = get_custom_columns()

	#Visualization
	#Univariate
	show_histogram(df)
	show_density_plots(df)
	show_whisker_plots(df)

	#Multivariate
	df_correlation: DataFrame = get_correlation(df)
	show_correlation_plot(df_correlation, columns)
	df_skew: DataFrame = get_skew(df)
	df_kurtosis: DataFrame = get_kurtosis(df)
	show_scatter_plot(df)

	#Target Variables
	df_class_distribution: DataFrame = get_class_distribution(df, "diabetes")
	df["diabetes"].value_counts().plot(kind="bar")
	pyplot.title("Target Variable (0=Negative Diabets, 1= Positive Diabetes)")
	show_plot()

	positive = df_class_distribution.iloc[1]
	negative = df_class_distribution.iloc[0]
	total = positive + negative
	pyplot.title("Diabetes class distribution")
	pyplot.pie([positive, negative], labels=[f"Positive ({positive/total * 100:.2f}%)", f"Negative ({negative/total * 100:.2f}%)"])
	show_plot()