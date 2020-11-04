
# Cross Validation Classification LogLoss


from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression

def main():
    PATH = "../pima-indians-diabetes.data.csv"
    columns = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
    df = read_csv(PATH, names=columns)
    array = df.values
    X = array[:,0:8]
    Y = array[:,8]	
    kfold = KFold(n_splits=10, random_state=7, shuffle=True)
    model = LogisticRegression(solver="lbfgs", max_iter=1000)
    results = cross_val_score(model, X, Y, cv=kfold, scoring="neg_log_loss")
    print(f"Logloss mean:{results.mean():.3f}%")
    print(f"Lgloss STD: {results.std():.3f}%")

if __name__ == "__main__":
	main()