import numpy
import pandas

def run_series():
	myarray = numpy.array([1, 2, 3])
	rownames = ['a', 'b', 'c']
	myseries = pandas.Series(myarray, index=rownames)
	return myseries

# series = run_series()
# print(series)
# print(type(series))
# print(series[0])
# print(series["a"])



def run_data_frame():
	myarray = numpy.array([[1, 2, 3], [4, 5, 6]])
	rownames = ['a', 'b']
	colnames = ['one', 'two', 'three']
	mydataframe = pandas.DataFrame(myarray, index=rownames, columns=colnames)
	return mydataframe


data_frame = run_data_frame()
print(data_frame)
print(type(data_frame))
print(data_frame["one"])
print(data_frame.two)