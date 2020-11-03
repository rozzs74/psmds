from pandas import read_csv
from numpy import set_printoptions
from sklearn.preprocessing import MinMaxScaler


def main():
    columns = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
    df = read_csv("../pima-indians-diabetes.csv", names=columns)
    array = df.values
    X = array[:, 0 : 8] #0 -7
    Y = array[:, 8] # Last
    scaler = MinMaxScaler(feature_range=(0, 1))
    rescaledX = scaler.fit_transform(X)
    set_printoptions(precision=3)
    print(rescaledX)
if __name__ == "__main__":
    main()