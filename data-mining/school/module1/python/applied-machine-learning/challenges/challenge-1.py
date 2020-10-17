def get_gz(z, e):
    gz = (1/(1+e**-z))
    return gz
a = get_gz(8, 2.718)
b = get_gz(2, 2.718)
 
print(a)

print(b)

