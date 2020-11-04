
# Random Search Parameter Tuning

import numpy
from pandas import read_csv
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import Ridge
from scipy.stats import uniform

def main():
    PATH = "../pima-indians-diabetes.data.csv"
    columns = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
    df = read_csv(PATH, names=columns)
    array = df.values
    X = array[:,0:8]
    Y = array[:,8]
    param_grid = {"alpha": uniform()}
    model = Ridge()
    rsearch = RandomizedSearchCV(estimator=model, param_distributions=param_grid, n_iter=100, random_state=7)
    rsearch.fit(X, Y)
    print(rsearch.best_score_)
    print(rsearch.best_estimator_.alpha)
if __name__ == "__main__":
	main()