import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

PATH = "../../python/multi-linear-regression/data-sets.csv"

class MultiRegression():
	def __init__(self, path):
		self.path = path
	def train(self, data_frames):
		df = data_frames.fillna(0)
		label_encoder = LabelEncoder()
		Y = df["Score"]
		X = df['Country or region'] = df['Country or region'].apply(lambda x: np.where(df['Country or region'].unique()==x)[0][0])
		X = df[['GDP per capita','Social support', 'Healthy life expectancy', 'Freedom to make life choices', 'Generosity', 'Perceptions of corruption', "Country or region"]] 
		regr = linear_model.LinearRegression()
		regr.fit(X, Y)
		print('Intercept: \n', regr.intercept_)
		print('Coefficients: \n', regr.coef_)
		New_GDP = 0.916
		New_Social = 1.078
		New_Health = 0.612
		New_Freedom = 0.411
		New_Generosity = 0.219
		New_Corruption = 0.125	
		New_Country = 41
		print ('Predicted Score: \n', regr.predict([[New_GDP, New_Social, New_Health, New_Freedom, New_Generosity, New_Corruption, New_Country]]))
		X = sm.add_constant(X) 
		model = sm.OLS(Y, X).fit()
		predictions = model.predict(X) 
		print_model = model.summary()
		print(print_model)
		print(f"r2_score: {r2_score(Y, predictions)}")
		print(f" mean_squared_error: {mean_squared_error(Y, predictions)}")
		print(f"mean_absolute_error: {mean_absolute_error(Y, predictions)}")
		

	def read(self):
		data_frames = pd.read_csv(PATH)
		return data_frames
	
model = MultiRegression(PATH)

df = model.read()
# df.info()
columns = df.columns
# keys = df.keys()
model.train(df)
