
from pandas import read_csv
from pandas import set_option
from matplotlib import pyplot
from pandas.plotting import scatter_matrix
import numpy

def main():
	path = "../../machine-learning-mastery-with-Python/pima-indians-diabetes.data.csv"

	columns = ["preg", "plas", "pres", "skin", "test", "mass", "pedi", "age", "class"]
	df = read_csv(path, names=columns)
	# show_histogram(df)
	# show_density_plots(df)
	# show_whisker_plots(df)
	# df_corr = get_correlation(df)
	# show_correlation_plot(df_corr, columns)
	show_scatter_plot(df)


def show_histogram(df):
	df.hist()
	show_plot()

def show_density_plots(df):
	df.plot(kind="density", subplots=True, layout=(3, 3), sharex=False)
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

def show_scatter_plot(df):
	scatter_matrix(df)
	show_plot()

def show_plot():
	pyplot.show()

def get_correlation(df):
	return df.corr()


if __name__ == "__main__":
	main()	