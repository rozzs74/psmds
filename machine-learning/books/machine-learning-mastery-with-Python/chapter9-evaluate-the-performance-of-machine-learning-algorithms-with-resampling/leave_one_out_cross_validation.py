
#  Evaluate using Leave One Out Cross Validation

from pandas import read_csv
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression

def main():
    PATH = "../pima-indians-diabetes.data.csv"
    columns = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
    df = read_csv(PATH, names=columns)
    array = df.values
    X = array[:,0:8]
    Y = array[:,8]	
    num_folds = 10
    loocv = LeaveOneOut()
    model = LogisticRegression(solver="lbfgs", max_iter=1000)
    results = cross_val_score(model, X, Y, cv=loocv)
    print(f"Accuracy mean:{results.mean()*100:.3f}%")
    print(f"Accuracy STD: {results.std()*100:.3f}%")
if __name__ == "__main__":
	main()