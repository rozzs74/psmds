
from pandas import read_csv
from pandas import set_option

def main():
	path = "../../machine-learning-mastery-with-Python/pima-indians-diabetes.data.csv"

	columns = ["preg", "plas", "pres", "skin", "test", "mass", "pedi", "age", "class"]
	df = read_csv(path, names=columns)
	head = get_peek(df, 20)
	# Data profiling
	dimension = get_dimension(df)
	df_descriptive_statistics = get_descriptive_statistics(df)
	df_info = get_data_information(df)
	df_data_types = get_data_types(df)
	# End Data Profiling
	df_class_distribution = get_class_distribution(df, "class")
	df_correlation = get_correlation(df, "pearson")
	df_skew = get_skew(df)
	print(df_skew)

def get_peek(df, n):
	return df.head(n)

def get_dimension(df):
	return df.shape

def get_data_types(df):
	return df.dtypes

def get_data_information(df):
	return df.info()

def get_descriptive_statistics(df):
	run_option()
	return df.describe()

def get_class_distribution(df, name):
	return df.groupby(name).size()

def get_correlation(df, method):
	run_option()
	return df.corr(method=method)

def get_skew(df):
	return df.skew()

def run_option():
	set_option("display.width", 100)
	set_option("precision", 3)


if __name__ == "__main__":
	main()	