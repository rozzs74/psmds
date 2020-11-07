# Load libraries
import numpy
from matplotlib import pyplot
from pandas import read_csv
from pandas import set_option
from pandas import concat
from pandas.plotting import scatter_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis 
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier


def main():
	PATH = "../sonar.csv"
	# Analyze Data (Data Profiling, Imputation strategies, and Visualization)
	df = read_csv(PATH, header=None)
	df_head = get_peek(df, 5)
	# df_tail = get_tail(df, 5)
	# df_dimension = get_dimension(df)
	# df_no_of_rows = df_dimension[0]
	# df_no_of_columns = df_dimension[1]
	# df_data_types = get_data_types(df)
	# # df_info = get_data_information(df)
	# df_descriptive_statistics = get_descriptive_statistics(df)
	# df_missing_value = get_missing_value(df)

	# Visualization
	# Unimodal Data Visualization
	# show_histogram(df)
	# show_density_plots(df)

	# Multimodal
	# df_correlation = get_correlation(df)
	# show_correlation_matrix(df_correlation)

	n_array = df.values
	X = n_array[:, 0 : 60].astype(float)
	Y = n_array[:, 60]
	X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=0.20, random_state=7)
	
def show_correlation_matrix(corr):
	fig = pyplot.figure()
	ax = fig.add_subplot(111)
	cax = ax.matshow(corr, vmin=-1, vmax=1, interpolation="none")
	fig.colorbar(cax)
	show_plot()

def get_peek(df, n):
	return df.head(n)

def get_tail(df, n):
	return df.tail(n)

def get_correlation(df):
	return df.corr()

def get_dimension(df):
	return df.shape

def get_missing_value(df):
	mask = df.isnull()
	total = mask.sum()
	percent = 100 * mask.mean()
	missing_value = concat([total, percent], axis=1, join="outer", keys=["count_missing", "percentage_missing"])
	missing_value.sort_values(by="percentage_missing", ascending=False, inplace=True)
	return missing_value

def get_data_types(df):
	return df.dtypes

def get_data_information(df):
	return df.info()

def get_descriptive_statistics(df):
	return df.describe()

def show_histogram(df):
	df.hist()
	show_plot()


def show_density_plots(df):
	df.plot(kind="density", subplots=True, layout=(8, 8), sharex=False)
	show_plot()


def show_plot():
	pyplot.show()

if __name__ == "__main__":
	main()