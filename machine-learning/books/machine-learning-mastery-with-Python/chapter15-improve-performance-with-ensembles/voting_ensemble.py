# Voting Ensemble

from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier

def main():
    PATH = "../pima-indians-diabetes.data.csv"
    columns = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
    df = read_csv(PATH, names=columns)
    array = df.values
    X = array[:,0:8]
    Y = array[:,8]
    kfold = KFold(n_splits=10, random_state=7, shuffle=True)
    models = [("lr", LogisticRegression(solver="lbfgs", max_iter=1000)), ("cart", DecisionTreeClassifier()), ("svm", SVC())]
    ensemble = VotingClassifier(models)
    results = cross_val_score(ensemble, X, Y, cv=kfold)
    print(round(results.mean()*100, 2))

if __name__ == "__main__":
	main()