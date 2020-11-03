# SVR

from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVR

def main():
    PATH = "../housing.csv"	
    columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
    df = read_csv(PATH, names=columns)
    array = df.values
    X = array[:,0:13]
    Y = array[:,13]
    kfold = KFold(n_splits=10, random_state=7, shuffle=True)
    model = SVR()
    results = cross_val_score(model, X, Y, cv=kfold, scoring="neg_mean_squared_error")
    print(f"MSE: {results.mean():.3f}%")

if __name__ == "__main__":
	main()