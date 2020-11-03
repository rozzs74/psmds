# Feature Extraction with PCA
from pandas import read_csv
from sklearn.decomposition import PCA

def main():
	PATH = "../pima-indians-diabetes.data.csv"
	columns = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
	df = read_csv(PATH, names=columns)
	array = df.values
	X = array[:,0:8]
	Y = array[:,8]	
	
	# feature extraction 
	pca = PCA(n_components=3)
	fit = pca.fit(X)
	# summarize components
	print("Explained Variance", fit.explained_variance_ratio_ )
	print(fit.components_)

if __name__ == "__main__":
	main()