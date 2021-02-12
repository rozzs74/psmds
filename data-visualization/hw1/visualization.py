import streamlit as st
import seaborn as sns
import random
import numpy as np
from pandas import read_csv
from pandas import DataFrame
from pandas import concat

from matplotlib import pyplot
st.title("Exploratory Data Analysis")
st.header("Data profiling")

@st.cache
def load_data():
	df = read_csv("./DataVizCensus2020-AnonymizedResponses.csv")
	return df

st.subheader("Raw data:")
df = load_data()
df = df.drop_duplicates()
df = df.fillna(0)

st.subheader("Tail:")
df_tail = df.tail(5)
df_tail

st.subheader("Dimension:")
df_shape = df.shape
"No of rows: ", df_shape[0] , "No of columns: ", df_shape[1]

st.subheader("Columns:")
df.columns

# st.subheader("Data types:")
# df.dtypes
# print(df.dtypes)
# fig1 = pyplot.figure()
# ax = sns.heatmap(df.isnull(), cbar=False)
# st.pyplot(fig1)

# gender = df["gender_collapsed"]
# professional_exp = df["How many years of professional experience do you have?"]
# # professional_exp
# education_attain = df["What is the highest level of education you have completed?"]
# a = df["How did you learn data visualization?"]
# # education_attain
# b = df["How many years of data visualization experience do you have doing professional data visualization?"]
# # c = df[" What technologies do you use to visualize data?"]
c = df["What country do you live in?"]
c
d = df["How many years of data visualization experience do you have doing professional data visualization?"]
x = d.replace(["less than 1", "(16-20]", "(5-10]", "(11-15]", "(21-25]", "(26-30]"], [0.5, random.randint(16, 20), random.randint(5, 10), random.randint(11, 15), random.randint(21, 25), random.randint(26, 30)]).astype(int)

countries = ["United States", "Singapore", "United Kingdom", "Australia", "India"]

top_five = []

v = DataFrame({'country': c,'year_experience': x})

for country in countries:
	print(country)
	a = v.loc[v["country"] == country].sum()
	top_five.append(a.iloc[1])


y = DataFrame({"country": countries, "total years experience per country": top_five})
y
# a = v.loc[v["country"] == "United States"].sum()
# a[1]
# # v.iloc[169]
sns.barplot(x="country",  y="total years experience per country", data=y)
st.set_option('deprecation.showPyplotGlobalUse', False)
st.pyplot()
