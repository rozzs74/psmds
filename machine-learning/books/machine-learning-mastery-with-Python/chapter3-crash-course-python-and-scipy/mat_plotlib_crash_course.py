import matplotlib.pyplot as plt
import numpy


myarray = numpy.array([1, 2, 3])
def run_line_plot():
    plt.plot(myarray)
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.show()

# run_line_plot()

def run_scatter_plot():
    x = numpy.array([1, 2, 3])
    y = numpy.array([2, 4, 6])
    plt.scatter(x, y)
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.show()


run_scatter_plot()