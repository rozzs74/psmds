from pandas import read_csv
from pandas import set_option
from matplotlib import pyplot

def main():
    pass

def get_columns():
    return []

def get_peek(df):
	return df.head(df)

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

def run_option():
	set_option("display.width", 100)
	set_option("precision", 3)

def get_correlation(df):
	return df.corr()

def get_skew(df):
	return df.skew()

def get_kurtosis(df):
    return df.kurtosis()

def impute_missing_value(df, num):
	return df.fillna(num)

if __name__ == "__main__":
	PATH = "./fish.csv"
	main()