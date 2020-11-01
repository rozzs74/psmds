from pandas import read_csv
from pandas import set_option
from pandas import concat
from pandas.plotting import scatter_matrix
from collections import Counter
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

import missingno as msno
import json
import numpy
import seaborn as sns

def main():
	#Exploratory Data Analysis (Data profiling, Imputation strategies and Visualization)

	# Data profiling
	columns = get_columns()
	df = read_csv(PATH, names=columns)
	df_head = get_peek(df, 5)
	print(df_head)
	df_tail = get_tail(df, 5)
	print(df_tail)
	df_variables_description = get_variable_description()
	print(json.dumps(df_variables_description, indent=4))
	df_missing_value_percentage = get_missing_value_percentage(df)
	print(df_missing_value_percentage)
	df_is_there_duplicates = check_duplicates(df)
	print(f"Is there duplicates {df_is_there_duplicates}")
	df_is_there_missing_value = check_missing_values(df)
	if df_is_there_missing_value == False:
		df_dimension = get_dimension(df)
		print(f"number of rows: {df_dimension[0]} \nnumber of columns: {df_dimension[1]}")
		df_data_types = get_data_types(df)
		print(df_data_types)
		df_information = get_data_information(df)
		df_descriptive_statistics = get_descriptive_statistics(df)
		print(df_descriptive_statistics)

		# Visualization
		# Univariate Plots
		# show_histogram(df)
		# show_density_plots(df)
		# show_whisker_plots(df)

		# Multivariate plots
		df_stash_columns = ["Weight", "Standard_Length", "Fork_Length", "Total_Length", "Height", "Width"]
		# df_correlation = get_correlation(df)
		# print(df_correlation)
		# show_correlation_plot(df_correlation, df_stash_columns)
		# df_skew = get_skew(df)
		# print(df_skew)
		# df_kurtosis = get_kurtosis(df)
		# print(df_kurtosis)
		# show_scatter_plot(df)

		# Categorical Variables
		# fish_type = df["Type"].value_counts()
		# show_bar_plot(fish_type, "Counts of Species")

		# Data preparation 
		df_outliers = df.loc[get_outliers(df, df_stash_columns)]
		print(df_outliers)
		df = drop_items(df, [142, 143, 144])
		# print(df)
		m_vars = get_training_variables(df, "Weight")
		y = m_vars["y"]
		X = m_vars["X"]
		X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
		model_fit = train_model(X_train, y_train)
		intercept = model_fit.intercept_
		coefficent = model_fit.coef_
		print('Model intercept: ', intercept)
		print('Model coefficients: ', coefficent)
		y_head = predict(model_fit, X_train)
		y_pred = predict(model_fit, X_test)
		show_regression_formula(intercept, coefficent)
		show_summary(X_train, y_train, model_fit, "Modeling Accuracy results (r2_score):")
		show_summary(X_test, y_test, model_fit, "Prediction Accuracy results (r2_score):")

		pyplot.scatter(X_test["Total_Length"], y_test, color="green", alpha=0.4)
		pyplot.scatter(X_test["Total_Length"], y_pred, color="blue", alpha=0.4)
		pyplot.xlabel("Total Length in cm")
		pyplot.ylabel("Weight of the fish")
		pyplot.title("Multi Linear Regression for Fish Weight Estimation")
		show_plot()

		pyplot.scatter(X_test["Height"], y_test, color="orange", alpha=0.5)
		pyplot.scatter(X_test["Height"], y_pred, color="blue", alpha=0.5)
		pyplot.xlabel("Height in cm")
		pyplot.ylabel("Weight of the fish")
		pyplot.title("Multi Linear Regression for Fish Weight Estimation")
		show_plot()

		pyplot.scatter(X_test['Width'], y_test, color='gray', alpha=0.5)
		pyplot.scatter(X_test['Width'], y_pred, color='red', alpha=0.5)
		pyplot.xlabel('Width in cm')
		pyplot.ylabel('Weight of the fish')
		pyplot.title("Multi Linear Regression for Fish Weight Estimation")
		show_plot()
	else:
		# Impute startegy here
		pass

def show_regression_formula(intercept, coefficent):
	print('y = ' + str('%.2f' % intercept) + ' + ' + str('%.2f' % coefficent[0]) + '*X1 ' + str('%.2f' % coefficent[1]) + '*X2 ' + str('%.2f' % coefficent[2]) + '*X3 + ' + str('%.2f' % coefficent[3]) + '*X4 + ' + str('%.2f' %coefficent[4]) + '*X5')

def show_summary(x, y, model, title):
	print(f"{title} {format(model.score(x, y) * 100, '.2f')}%")

def train_model(x, y):
	linear_regression = LinearRegression()
	linear_regression.fit(x, y)
	return linear_regression

def predict(model, x):
	return model.predict(x)

def get_summary(y_train, y_pred):
	pass

def get_training_variables(df, feature):
	return {"y": df[feature], "X": df.iloc[:, 2: 7]}

def get_columns():
    return ["Type", "Weight", "Standard_Length", "Fork_Length", "Total_Length", "Height", "Width"]

def get_variable_description():
	return {"Type": "The species of fish.", "Weight": "Weight of fish in grams.", "Standard_Length": "Standard length of fish in cm.", "Fork_Length": "Fork length of fish in cm.", "Total_Length": "Total length of fish in cm.", "Height": "Body dept (height) of fish in cm.", "Width": "Body Thickness (width) of fish in cm."}

def get_peek(df, n):
	return df.head(n)

def get_tail(df, n):
	return df.tail(n)

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
	set_option("precision", 2)

def get_correlation(df):
	return df.corr()

def get_skew(df):
	return df.skew()

def get_kurtosis(df):
    return df.kurtosis()

def get_outliers(df,features):
	outlier_indices = []
	first_percentile = 25
	last_percentile = 75

	for columns in features:
		q1 = numpy.percentile(df[columns], first_percentile)
		q3 = numpy.percentile(df[columns], last_percentile)
		q_diff = q3 - q1
		outlier_stepper = q_diff * 1.5
		outliers = df[(df[columns] < q1 - outlier_stepper) | (df[columns] > q3 + outlier_stepper)].index
		outlier_indices.extend(outliers)

	outlier_indices = Counter(outlier_indices)
	multiple_outliers = list(k for k, i in outlier_indices.items() if i > 2)
	return multiple_outliers

def get_missing_value(df):
	mask = df.isnull()
	total = mask.sum()
	percent = 100 * mask.mean()
	missing_value = concat([total, percent], axis=1, join="outer", keys=["count_missing", "percentage_missing"])
	missing_value.sort_values(by="percentage_missing", ascending=False, inplace=True)
	return missing_value

def get_missing_value_percentage(df):
	return round(100 * (df.isnull().sum() / len(df)), 2)

def check_duplicates(df):
	stash = df.copy()
	if stash.shape == df.shape:
		return False
	else:
		return True

def check_missing_values(df):
	is_missing = df.isnull().values.any()
	print(str("Is there any NaN values in the dataset?"), is_missing)
	return is_missing

def drop_items(df, to_drop):
	stash = df.drop(to_drop)
	return stash

def impute_missing_value(df, num):
	return df.fillna(num)

# Missingness
def show_missing_plot(df):
	null = df.isnull()
	null_columns = df.columns[null.any()].tolist()
	msno.heatmap(df[null_columns], figsize=(18, 18))
	show_plot()

def show_missing_heat_map(df):
	mask = df.isnull()
	nullable_columns = df.columns[mask.any()].tolist()
	msno.heatmap(df[nullable_columns], figsize=(18, 18))
	show_plot()

# Distribution
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

def show_bar_plot(df, title):
	df.plot(kind="bar", title=title)
	show_plot()

def show_plot():
	pyplot.show()

def show_correlation_plot(correlations, names):
	fig = pyplot.figure()
	ax = fig.add_subplot(111)
	cax = ax.matshow(correlations, vmin=-1, vmax=1)
	fig.colorbar(cax)
	ticks = numpy.arange(0, 6, 1)
	ax.set_xticks(ticks)
	ax.set_yticks(ticks)
	ax.set_xticklabels(names) 
	ax.set_yticklabels(names)
	show_plot()

if __name__ == "__main__":
	PATH = "./fish.csv"
	main()