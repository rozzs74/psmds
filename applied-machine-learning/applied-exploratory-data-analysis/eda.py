from pandas import read_csv
from pandas import set_option
from pandas import concat
from pandas.plotting import scatter_matrix
from matplotlib import pyplot

import missingno as msno
import numpy

def main():
	columns = get_columns()
	df = read_csv(PATH, names=columns)
	df_head = get_peek(df, 10)
	# # print(df_head)
	# df_dimension = get_dimension(df)
	# # print(df_dimension)
	# df_data_types = get_data_types(df)
	# df_correlation = get)df_correlation(df)
	# print(df_correlation)
	# # print(df_data_types)
	# df_information = get_data_information(df)
	# print(df_information)
	# df_descriptive_statistics = get_descriptive_statistics(df)
	# print(df_descriptive_statistics)
	# df_missing_value = get_missing_value(df)
	# print(df_missing_value)
	# missing_plot(df)
	# missing_heat_map(df)
	new_df = impute_missing_value(df)
	# new_df_head = get_peek(new_df, 20)

	# new_df.diabetes.value_counts().plot(kind="bar")
	# pyplot.title("Target Variables(0=Negative Diabetes, 1=Positive Diabetes)")
	# show_plot()

	numerical_values = new_df.select_dtypes(include=[numpy.number])



	new_df_class_distribution = get_class_distribution(new_df, "diabetes")
	positive = new_df_class_distribution.iloc[0]
	negative = new_df_class_distribution[1]
	pyplot.title("Diabetes class distribution")
	pyplot.pie([positive, negative], labels=["Postive", "Negative"])
	show_plot()



	# numerical_values.nunique().sort_values()
	# show_density_plots(numerical_values)
	show_histogram(numerical_values)

	
	new_df_skew = get_skew(new_df)
	new_df_corr = get_correlation(new_df)
	show_correlation_plot(new_df_corr, columns)

	show_scatter_plot(new_df)
	show_whisker_plots(new_df)

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