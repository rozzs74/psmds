from sklearn.preprocessing import StandardScaler
from pandas import read_csv
from numpy import set_printoptions

def main():
	PATH = "../pima-indians-diabetes.data.csv"
	columns = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
	df = read_csv(PATH, names=columns)
	array = df.values
	X = array[:, 0 : 8] # Column 1 - 7
	Y = array[:, 8] # Column 8
	scaler = StandardScaler().fit(X) # Prepare independent variables for transformation
	rescaledX = scaler.transform(X) # ready to use in modeling
	set_printoptions(precision=3)
	print(rescaledX[0: 5,:])

if __name__ == "__main__":
	main()