from sklearn.preprocessing import Binarizer
from pandas import read_csv
from numpy import set_printoptions


def main():
	PATH = "../pima-indians-diabetes.data.csv"
	columns = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
	df = read_csv(PATH, names=columns)
	array = df.values
	X = array[:, 0: 8]
	Y = array[:, 8]
	binarizer = Binarizer(threshold=0.0).fit(X)
	binaryX = binarizer.transform(X)
	set_printoptions(precision=3)
	print(binaryX[0: 5,:])
	
if __name__ == "__main__":
	main()