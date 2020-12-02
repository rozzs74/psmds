import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import graphviz

from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz

# Step 1 Read data set
df = pd.read_csv("iris-data.csv")

# Step 2 Generate random seed for testing
np.random.seed(10)
samples = np.random.randint(0, len(df), 10)
df_test = df.iloc[samples]
df = df.drop(samples)

#Step 3 Fit the model for the training data and then check the accuracy
model = DecisionTreeClassifier()
model = model.fit(df[['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width']], df.Species)

# Step 4 check the performance by knowing the score from test set
print(model.score(df[['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width']], df.Species)) #1.0

dot_data = export_graphviz(model, out_file=None)
graph = graphviz.Source(dot_data)