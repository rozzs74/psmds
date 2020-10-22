import numpy
from numpy import loadtxt


path = "../../machine-learning-mastery-with-Python/pima-indians-diabetes.data.csv"
def read(path):
	content = open(path, "r")
	data = loadtxt(content, delimiter=",")
	print(data)
	print(data.shape)



	
read(path)