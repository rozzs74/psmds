
from pandas import read_csv
from pandas import set_option
from matplotlib import pyplot

def main():
	path = "../../machine-learning-mastery-with-Python/pima-indians-diabetes.data.csv"

	columns = ["preg", "plas", "pres", "skin", "test", "mass", "pedi", "age", "class"]
	df = read_csv(path, names=columns)
	# show_histogram(df)
	# show_density_plots(df)
	show_whisker_plots(df)


def show_histogram(df):
	df.hist()
	show_plot()

def show_density_plots(df):
	df.plot(kind="density", subplots=True, layout=(3, 3), sharex=False)
	show_plot()

def show_whisker_plots(df):
	df.plot(kind="box", subplots=True, layout=(3, 3), sharex=False, sharey=False)
	show_plot()

def show_plot():
	pyplot.show()


if __name__ == "__main__":
	main()	