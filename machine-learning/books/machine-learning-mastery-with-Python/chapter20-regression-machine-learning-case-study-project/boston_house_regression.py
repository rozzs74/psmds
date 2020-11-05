# Load libraries
import numpy
from numpy import arange
from matplotlib import pyplot
from pandas import read_csv
from pandas import set_option
from pandas.plotting import scatter_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.linear_model import ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import mean_squared_error

def main():
	PATH = "../housing.csv"
	columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
	df = read_csv(PATH, names=columns)
	df_dimension = df.shape
	df_data_types = df.dtypes
	df_head = df.head(5)
	df_descriptive_statistics = df.describe()
	df_corr = df.corr(method="pearson")
	set_option("precision", 2)

	# Unimodal Visualization
	# df.hist(sharex=False, sharey=False, xlabelsize=1, ylabelsize=1)
	# pyplot.show()

	# Density
	# df.plot(kind="density", subplots=True, layout=(4,4), sharex=False, legend=False, fontsize=1)
	# pyplot.show()

	# Box and whisker plots
	# df.plot(kind="box", subplots=True, layout=(4, 4), sharex=False, sharey=False, fontsize=8)
	# pyplot.show()

	# Multimodal Visualization
	scatter_matrix(df)
	pyplot.show()
if __name__ == "__main__":
	main()