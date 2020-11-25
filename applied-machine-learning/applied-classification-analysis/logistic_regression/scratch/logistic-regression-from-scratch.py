import numpy as np
import pandas as pd
from numpy.random import rand
from numpy import log, dot, e

def main():
	df = pd.read_csv(PATH, names=["pregnant", "glucose", "pressure", "triceps", "insulin", "mass", "pedigree", "age", "diabetes"])
	X = df.values[:, 0: 8]
	y = df.values[:, 8]

def compute_sigmoid(z):
	sigmoid = 1 / (1 + np.exp(-z))
	return sigmoid

def get_z(x, weights):
	return np.dot(x, weights)

def cost_function(self, X, y, weights):                 
	z = dot(X, weights)
	predict_1 = y * log(self.sigmoid(z))
	predict_0 = (1 - y) * log(1 - self.sigmoid(z))
	return -sum(predict_1 + predict_0) / len(X)
    
def fit(self, X, y, epochs=25, lr=0.05):        
	loss = []
	weights = rand(X.shape[1])
	N = len(X)
                 
	for _ in range(epochs):        
		# Gradient Descent
		y_hat = self.sigmoid(dot(X, weights))
		weights -= lr * dot(X.T,  y_hat - y) / N            
		# Saving Progress
		loss.append(self.cost_function(X, y, weights)) 
            
	self.weights = weights
	self.loss = loss

def predict(self, X):        
	# Predicting with sigmoid function
	z = dot(X, self.weights)
	# Returning binary result
	return [1 if i > 0.5 else 0 for i in self.sigmoid(z)]

if __name__ == "__main__":
	PATH = "../../pima-indians-diabetes.data.csv"
	main()