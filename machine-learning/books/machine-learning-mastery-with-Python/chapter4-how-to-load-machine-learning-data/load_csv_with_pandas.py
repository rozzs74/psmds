from pandas import read_csv

# path = "../../pima-indians-diabetes.data.csv"
path = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"

def read(path):
	names = ["preg", "plas", "pres", "skin", "test", "mass", "pedi", "age", "class"]
	data = read_csv(path, names=names)
	print(data)
	print(data.shape)


read(path)