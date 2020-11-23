import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler


# Step 1 Load the data
df = pd.read_excel("energy.xlsx")

# Step 2 Define dependent and indepdent variable then standardize independent variables
X = df.iloc[:,:4]
y = df.iloc[:, -1]
X = StandardScaler().fit_transform(X)

# Step 3  Define MSE
def cost_function(X, Y, B):
	m = len(Y)
	J = np.sum((X.dot(B) - Y) ** 2)/(2 * m)
	return J

# Step 4 Batch Gradient Descent
def batch_gradient_descent(X, Y, B, alpha, iterations):
	cost_history = [0] * iterations
	m = len(Y)
	for i in range(iterations):
		# Hypothesis values
		h = X.dot(B)
		# Difference between H and Actual Y
		loss = h - Y
		# Gradient Computation
		gradient = X.T.dot(loss) / m
		# Change the value of B using Gradient
		B = B - alpha * gradient
		# New cost value
		cost = cost_function(X, Y, B)
		cost_history[i] = cost
	return B, cost_history


def r2(y_,y):
	sst = np.sum((y-y.mean())**2)
	ssr = np.sum((y_-y)**2)
	r2 = 1-(ssr/sst)
	return(r2)


# Step 4 Split train and test data sets
m = 7000
f = 2
X_train = X[:m,:f]
X_train = np.c_[np.ones(len(X_train),dtype="int64"),X_train]
y_train = y[:m]
X_test = X[m:,:f]
X_test = np.c_[np.ones(len(X_test),dtype='int64'),X_test]
y_test = y[m:]

# Step 5 Train the model
B = np.zeros(X_train.shape[1])
alpha = 0.005
iter_ = 2000

newB, cost_history = batch_gradient_descent(X_train, y_train, B, alpha, iter_)

# Step 9 Compare training history and errors 
plt.figure(figsize=(10, 7))
plt.plot(cost_history);
plt.title('Training History');
plt.xlabel('epoch');
plt.ylabel('Error');
plt.show()

# Step 10 Make Prediction
y_pred = X_test.dot(newB)
print(r2(y_pred, y_test))