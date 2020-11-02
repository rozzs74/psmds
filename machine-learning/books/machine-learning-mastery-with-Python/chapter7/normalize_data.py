from sklearn.preprocessing import Normalizer
from pandas import read_csv
from numpy import set_printoptions


def main():
	PATH = "../pima-indians-diabetes.data.csv"
	columns = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
	df = read_csv(PATH, names=columns)
	array = df.values
	X = array[:, 0: 8]
	Y = array[:, 8]
	scaler = Normalizer().fit(X)
	normalizedX = scaler.transform(X)
	set_printoptions(precision=3)
	print(normalizedX[0: 5,:])
	
if __name__ == "__main__":
	main()