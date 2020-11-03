from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression


def main():
	PATH = "../housing.csv"	
	columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
	df = read_csv(PATH, names=columns)
	array = df.values
	X = array[:, 0 : 13]
	Y = array[:, 13]
	kfold = KFold(n_splits=10, random_state=7, shuffle=True)
	model = LinearRegression()
	scoring = "r2"
	results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
	print(f"R^2 Mean {results.mean():.3f}")
	print(f"R^2 STD  {results.std():.3f}")


if __name__ == "__main__":
	main()