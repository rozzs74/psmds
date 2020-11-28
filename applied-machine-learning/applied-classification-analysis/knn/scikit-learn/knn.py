import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier as KNN



# Step 1 Read data
df = pd.read_csv("../iris-data.csv")
head = df.head(5)

# Step 2 Visualize Class

df_test = df.iloc[134] # Drop this
df = df.drop([134])
markers = {
    'Iris-setosa': {'marker': 'x', 'facecolor': 'k', 'edgecolor': 'k'},
    'Iris-versicolor': {'marker': '*', 'facecolor': 'none', 'edgecolor': 'k'},
    'Iris-virginica': {'marker': 'o', 'facecolor': 'none', 'edgecolor': 'k'},
}
plt.figure(figsize=(10, 7))
for name, group in df.groupby('Species'):
    plt.scatter(group['Sepal Length'], group['Petal Width'], 
                label=name,
                marker=markers[name]['marker'],
                facecolors=markers[name]['facecolor'],
                edgecolor=markers[name]['edgecolor'])
    
plt.title('Species Classification Sepal Length vs Petal Width');
plt.xlabel('Sepal Length (mm)');
plt.ylabel('Petal Width (mm)');
plt.legend();
plt.show()

model = KNN(n_neighbors=3)
model.fit(X=df[['Petal Width', 'Sepal Length']], y=df.Species)
model.score(X=df[['Petal Width', 'Sepal Length']], y=df.Species)
model.predict(df_test[['Petal Width', 'Sepal Length']].values.reshape((-1, 2)))[0]
df.iloc[134].Species