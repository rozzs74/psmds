# 1) Define the Problem
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
	# 2) Analyze the Data
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
	# scatter_matrix(df)
	# pyplot.show()
	
	# fig = pyplot.figure()
	# ax = fig.add_subplot(111)
	# cax = ax.matshow(df_corr, vmin=-1, vmax=1, interpolation="none")
	# fig.colorbar(cax)
	# ticks = numpy.arange(0, 14, 1)
	# ax.set_xticks(ticks)
	# ax.set_yticks(ticks)
	# ax.set_xticklabels(columns)
	# ax.set_yticklabels(columns)
	# pyplot.show()

	# 3) Prepare the Data
	# Split-out validation dataset
	array = df.values
	X = array[:,0:13]
	Y = array[:,13]
	X_train, X_test, Y_train, Y_test =  train_test_split(X, Y, test_size=0.20, random_state=7)

	# 4) Evaluate Algorithms
	# Spot-check algorithm baseline
	# models = [("LR", LinearRegression()), ("LASSO", Lasso()), ("EN", ElasticNet()), ("KNN", KNeighborsRegressor()), ("CART", DecisionTreeRegressor()), ("SVR", SVR())]
	# i = 0
	# model_results = []
	# model_name = []
	# while i < len(models):
	# 	el = models[i]
	# 	kfold = KFold(n_splits=10, random_state=7, shuffle=True)
	# 	cv_results = cross_val_score(el[1], X_train, Y_train, cv=kfold, scoring="neg_mean_squared_error")
	# 	model_results.append(cv_results)
	# 	model_name.append(el[0])
	# 	print(el[0])
	# 	print(f"Mean Estimated Accuracy: {cv_results.mean()}")
	# 	print(f"STD: {cv_results.std()}")
	# 	i += 1
	# 	if i == len(models):
	# 		fig = pyplot.figure()
	# 		fig.suptitle("Algorithm Comparison")
	# 		ax = fig.add_subplot(111)
	# 		pyplot.boxplot(model_results)
	# 		ax.set_xticklabels(model_name)
	# 		pyplot.show()

	# Spot-check algorithm with Standardization
	# pipelines = [
	# 	('ScaledLR', Pipeline([('Scaler', StandardScaler()),('LR', LinearRegression())])),
	# 	('ScaledLASSO', Pipeline([('Scaler', StandardScaler()),('LASSO',Lasso())])),
	# 	('ScaledEN', Pipeline([('Scaler', StandardScaler()),('EN', ElasticNet())])),
	# 	('ScaledKNN', Pipeline([('Scaler', StandardScaler()),('KNN', KNeighborsRegressor())])),
	# 	('ScaledCART', Pipeline([('Scaler', StandardScaler()),('CART', DecisionTreeRegressor())])),
	# 	('ScaledSVR', Pipeline([('Scaler', StandardScaler()),('SVR', SVR())]))
	# ]
	# model_results = []
	# model_name = []
	# i = 0
	# while i < len(pipelines):
	# 	el = pipelines[i]
	# 	kfold = KFold(n_splits=10, random_state=7, shuffle=True)
	# 	cv_results = cross_val_score(el[1], X_train, Y_train, cv=kfold, scoring="neg_mean_squared_error")
	# 	model_results.append(cv_results)
	# 	model_name.append(el[0])
	# 	print(f"name {el[0]} score: {cv_results.mean()} {cv_results.std()}")
	# 	i += 1
	# 	if i == len(pipelines):
	# 		fig = pyplot.figure()
	# 		fig.suptitle("Algorithm Comparison")
	# 		ax = fig.add_subplot(111)
	# 		pyplot.boxplot(model_results)
	# 		ax.set_xticklabels(model_name)
	# 		pyplot.show()
	# 		break

	# # Improve results with Tuning
	# scaler = StandardScaler().fit(X_train)
	# rescaledX = scaler.transform(X_train)
	# k_values = numpy.array([1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21])
	# param_grid = dict(n_neighbors=k_values)
	# model = KNeighborsRegressor()
	# kfold = KFold(n_splits=10, random_state=7, shuffle=True)
	# grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring="neg_mean_squared_error", cv=kfold)
	# grid_result = grid.fit(rescaledX, Y_train)
	# print(f"Best k {grid_result.best_score_} {grid_result.best_params_}")
	# means = grid_result.cv_results_["mean_test_score"]
	# stds = grid_result.cv_results_["std_test_score"]
	# params = grid_result.cv_results_["params"]
	# for mean, stdev, param in zip(means, stds, params):
	# 	print(f"mean={mean} stdev={stdev} param={param}")

	# Ensemble
	# ensembles = [
	# 	('ScaledAB', Pipeline([('Scaler', StandardScaler()),('AB',AdaBoostRegressor())])),
	# 	('ScaledGBM', Pipeline([('Scaler', StandardScaler()),('GBM',GradientBoostingRegressor())])),
	# 	('ScaledRF', Pipeline([('Scaler', StandardScaler()),('RF', RandomForestRegressor())])),
	# 	('ScaledET', Pipeline([('Scaler', StandardScaler()),('ET',ExtraTreesRegressor())]))
	# ]
	# ensemble_results = []
	# ensemble_name = []

	# i = 0
	# while i <= len(ensembles):
	# 	el = ensembles[i]
	# 	kfold = KFold(n_splits=10, random_state=7, shuffle=True)
	# 	cv_results = cross_val_score(el[1], X_train, Y_train, cv=kfold, scoring="neg_mean_squared_error")
	# 	ensemble_results.append(cv_results)
	# 	ensemble_name.append(el[0])
	# 	print(f"Name={el[0]} Mean={cv_results.mean()} STD={cv_results.std()}")
	# 	i += 1
	# 	if i == len(ensembles):
	# 		fig = pyplot.figure()
	# 		fig.suptitle("Algorithm Comparison Ensembles")
	# 		ax = fig.add_subplot(111)
	# 		pyplot.boxplot(ensemble_results)
	# 		ax.set_xticklabels(ensemble_name)
	# 		pyplot.show()
	# 		break
	# 5) Improve Results
	# Tune ensembe method
	# scaler = StandardScaler().fit(X_train)
	# rescaledX = scaler.transform(X_train)
	# param_grid = dict(n_estimators=numpy.array([10, 50, 100, 1000, 5000]))
	# model = ExtraTreesRegressor()
	# grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring="neg_mean_squared_error")
	# grid_result = grid.fit(rescaledX, Y_train)
	# print(f"Best score={grid_result.best_score_} param={grid_result.best_params_}")
	# means = grid_result.cv_results_["mean_test_score"]
	# std = grid_result.cv_results_["std_test_score"]
	# params = grid_result.cv_results_["params"]

	# for mean, stdev, param in zip(means, std, params):
		# print(f"mean={mean}, std={stdev}, params={param }")
	
	# Finalize model
	# 7) Present Results
	scaler = StandardScaler().fit(X_train)
	rescaledX = scaler.transform(X_train)
	model = ExtraTreesRegressor(random_state=7, n_estimators=1000)
	model.fit(rescaledX, Y_train)
	rescaledTestX = scaler.transform(X_test)
	predictions = model.predict(rescaledTestX)
	print(mean_squared_error(Y_test, predictions))
if __name__ == "__main__":
	main()