import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Step 1 Load the Data
df = pd.read_csv("housing_data.csv")
df.head()


# Step 2 Check Linearity between independent variables and dependent variables
fig = plt.figure(figsize=(10, 7))
fig.suptitle('Parameters vs Median Value')
ax1 = fig.add_subplot(121)
ax1.scatter(df.LSTAT, df.MEDV, marker='*', c='k');
ax1.set_xlabel('% lower status of the population')
ax1.set_ylabel('Median Value in $1000s')
ax2 = fig.add_subplot(122, sharey=ax1)
ax2.scatter(df.RM, df.MEDV, marker='*', c='k');
ax2.get_yaxis().set_visible(False)
ax2.set_xlabel('average number of rooms per dwelling');
plt.show()

# Step 3 Construct a model between indepent variables and dependent variables

model = LinearRegression()

# First linear relationship
model.fit(df.LSTAT.values.reshape((-1, 1)), df.MEDV.values.reshape((-1, 1)))
model.score(df.LSTAT.values.reshape((-1, 1)), df.MEDV.values.reshape((-1, 1)))


# Second linear relationship
model.fit(df.RM.values.reshape((-1, 1)), df.MEDV.values.reshape((-1, 1)))
model.score(df.RM.values.reshape((-1, 1)), df.MEDV.values.reshape((-1, 1)))

# Finalize them
model.fit(df[['LSTAT', 'RM']], df.MEDV.values.reshape((-1, 1)))
model.score(df[['LSTAT', 'RM']], df.MEDV.values.reshape((-1, 1)))