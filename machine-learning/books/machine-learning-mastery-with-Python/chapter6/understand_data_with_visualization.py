
from pandas import read_csv
from pandas import set_option
from matplotlib import pyplot

def main():
	path = "../../machine-learning-mastery-with-Python/pima-indians-diabetes.data.csv"

	columns = ["preg", "plas", "pres", "skin", "test", "mass", "pedi", "age", "class"]
	df = read_csv(path, names=columns)

	show_histogram(df)


def show_histogram(df):
	df.hist()
	show_plot()

def show_plot():
	pyplot.show()


if __name__ == "__main__":
	main()	