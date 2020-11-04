
# Grid Search Parameter Tuning
import numpy
from pandas import read_csv
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge
def main():
    PATH = "../pima-indians-diabetes.data.csv"
    columns = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
    df = read_csv(PATH, names=columns)
    array = df.values
    X = array[:,0:8]
    Y = array[:,8]
    alphas = numpy.array([1,0.1,0.01,0.001,0.0001,0])
    param_grid = dict(alpha=alphas)
    model = Ridge()
    grid = GridSearchCV(estimator=model, param_grid=param_grid)
    grid.fit(X, Y)
    print(grid.best_score_)
    print(grid.best_estimator_.alpha)
if __name__ == "__main__":
	main()