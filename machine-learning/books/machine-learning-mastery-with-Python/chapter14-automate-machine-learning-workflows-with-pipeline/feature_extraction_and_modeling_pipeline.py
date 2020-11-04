# Create a pipeline that standardizes the data then creates a model

from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
from sklearn.linear_model import LogisticRegression

def main():
    PATH = "../pima-indians-diabetes.data.csv"
    columns = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
    df = read_csv(PATH, names=columns)
    array = df.values
    X = array[:,0:8]
    Y = array[:,8]	

    # Create feature Union
    features = []
    features.append(("pca", PCA(n_components=3)))
    features.append(("select_best", SelectKBest(k=6)))
    feature_union = FeatureUnion(features)

    # Create pipeline
    pipeline = []
    pipeline.append(("feature_union", feature_union))
    pipeline.append(("standardize", StandardScaler()))
    pipeline.append(("logistic", LogisticRegression(solver="lbfgs", max_iter=1000)))
    
    model = Pipeline(pipeline)
    kfold = KFold(n_splits=10, random_state=7, shuffle=True)
    results = cross_val_score(model, X, Y, cv=kfold)
    print(results.mean() * 100)


if __name__ == "__main__":
	main()