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
	else:
		data_profiling(df)

def data_profiling(df):
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
def clean_data(df):
	pass

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

def main() -> None:
	# Understanding the Data
	df  = read_csv("./students.csv")
	understand_data(df)

if __name__ == "__main__":
	main()

