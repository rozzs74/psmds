from pandas import read_csv
from pandas import set_option
from pandas import concat
from pandas.plotting import scatter_matrix
from matplotlib import pyplot

import missingno as msno
import numpy

def main():
	pass

def get_columns():
	return ["pregnant", "glucose", "pressue", "triceps", "insulin", "mass", "pedigree", "age", "diabetes"]

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
	PATH = "./pima-indians-diabetes.data.csv"
	main()