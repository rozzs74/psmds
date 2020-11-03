import numpy

# Single Column
mylist = [1, 2, 3]
print(mylist)

myarray = numpy.array(mylist)
print(myarray)
print(type(myarray))
print(myarray.shape)
print(myarray[0:2])


# Multiple columns
multiple_list = [[1,2,3], [4,5,6]]
multiple_array = numpy.array(multiple_list)
print(f"First row {multiple_array[0]}")
print(f"Last row {multiple_array[-1]}")
print(f"Multiple rows {multiple_array[0:3]}")
print(f"Whole column {multiple_array[:, 2]}")

# Arithmetic
n1 = numpy.array([2, 2, 2])
n2 = numpy.array([3, 3, 3])
print(f"Sum {n1 + n2}")
print(f"Difference {n1 - n2}")
print(f"Product {n1 * n2}")
print(f"Qoutient {n1 / n2}")