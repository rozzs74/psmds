# Create a pipeline that standardizes the data then creates a model

from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


def main():
	PATH = "../pima-indians-diabetes.data.csv"
	columns = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
	df = read_csv(PATH, names=columns)
	array = df.values
	X = array[:,0:8]
	Y = array[:,8]	
	
	# Pipeline Creation
	pipeline = []
	pipeline.append(("standardize", StandardScaler()))
	pipeline.append(("lda", LinearDiscriminantAnalysis()))
	model = Pipeline(pipeline)

	# Pipeline Evaluation
	kfold = KFold(n_splits=10, random_state=7, shuffle=True)
	results = cross_val_score(model, X, Y, cv=kfold)
	print(results.mean() * 100)


if __name__ == "__main__":
	main()