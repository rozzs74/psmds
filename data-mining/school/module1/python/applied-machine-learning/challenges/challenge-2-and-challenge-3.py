
def get_gz(z, e):
    gz = (1/(1+e**-z))
    return gz



def main():
	z = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
	e = [1.718, 2.718, 3.718, 4.718, 5.718, 6.718, 7.718, 8.718, 9.718, 10.718]
	i = 0
	g = []
	while 0 < len(z):
		x = z[i]
		y = e[i]
		g.append(get_gz(x, y))
		i += 1
		if i == len(z):
			return g

g = main()