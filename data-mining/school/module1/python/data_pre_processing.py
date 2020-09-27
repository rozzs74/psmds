import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing

# PATH = "../python/tested_worldwide.csv"
PATH = "../python/depedtotalschoolenrollmentforpublicelementaryschools2005to2012_0.csv"

class DataPreprocessing():

	def __init__(self, path):
		self.path = path
	
	def clean(self, frames):
		""" This function is for cleaning the data frames, by filling all NULL values."""
		new_data_frames = frames.fillna(0)
		return new_data_frames
	def reduce(self, frames):
		print(frames)
		""" This function is for dropping duplicate values under data frames."""
		frames.drop_duplicates()
		return frames	
	def transform(self, frames):
		""" This function will transform the data into range value."""
		min_max_scaler = preprocessing.MinMaxScaler()
		frames[['Enrollment_2012']] = min_max_scaler.fit_transform(frames[['Enrollment_2012']])
		return frames
	def integrate(self, frames):
		""" This function simply create pie chart."""
		cols = []
		vals = []
		for col in frames.columns:
			if col[0] == "E":
				cols.append(col)
		for i in cols:
			vals.append(sum(frames[i]))
		print(vals)

		colors = ["red", "yellow", "white","blue","green", "orange", "pink", "yellowgreen"]
		plt.pie(vals, labels=cols, colors=colors,autopct='%1.1f%%', shadow=True, startangle=140)
		plt.axis('equal')
		plt.show()
		return cols
		# ax = plt.gca()
		# frames.plot(kind="scatter", x="daily_tested", y="Country_Region", color="blue")
		# plt.show()
	def read(self):
		""" This function read a CSV file."""
		data_sets = pd.read_csv(self.path, nrows=3000)
		return data_sets

# Index(['Date', 'Country_Region', 'Province_State', 'positive', 'active',
#        'hospitalized', 'hospitalizedCurr', 'recovered', 'death',
#        'total_tested', 'daily_tested', 'daily_positive'],
#       dtype='object')

job = DataPreprocessing(PATH)
output = job.read()
columns = output.columns
info = output.info()
stats = output.describe()
clean = job.clean(output)
# print(clean)
normalize = job.reduce(clean)
# print(normalize)
transform = job.transform(normalize)
integrate = job.integrate(normalize)

