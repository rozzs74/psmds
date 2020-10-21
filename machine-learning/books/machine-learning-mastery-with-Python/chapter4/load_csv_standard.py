import csv
import numpy

filename = "./pima-indians-diabetes.csv"

def read(file_name):
	with open(filename, "r") as content:
		print(content.read())


def read_csv(file_name):
	raw = open(filename, "r")
	reader = csv.reader(raw, delimiter=",", quoting=csv.QUOTE_NONE)
	list_content = list(reader)
	data = numpy.array(list_content)
	print(data.shape)

read_csv(filename)
read(filename)