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
st.set_option('deprecation.showPyplotGlobalUse', False)

@st.cache
def load_data():
	df = read_csv("./DataVizCensus2020-AnonymizedResponses.csv")
	return df

st.subheader("Raw data:")
df = load_data()
df = df.drop_duplicates()
df = df.fillna(0)
df

st.subheader("Tail:")
df_tail = df.tail(5)
df_tail

st.subheader("Dimension:")
df_shape = df.shape
"No of rows: ", df_shape[0] , "No of columns: ", df_shape[1]

st.subheader("Columns:")
df.columns

st.subheader("Data types:")
df.dtypes

st.subheader("Transformed:")
d = df["How many years of data visualization experience do you have doing professional data visualization?"]
x = d.replace(["less than 1", "(16-20]", "(5-10]", "(11-15]", "(21-25]", "(26-30]"], [0.5, random.randint(16, 20), random.randint(5, 10), random.randint(11, 15), random.randint(21, 25), random.randint(26, 30)]).astype(int)
new_df = DataFrame({
	"country": df["What country do you live in?"],
	"years_experience": x,
	"learn_method": df["How did you learn data visualization?"],
	"hours_in_data_engineering": df["Hours a week focused on data engineering?"],
	"hours_in_data_science": df["Hours a week focused on data science?"],
	"hours_in_data_prep_work": df["Hours a week focused on data prep work?"],
	"hours_in_design": df["Hours a week focused on design?"]
})
new_df

st.header("Data visualization")

def chart1(df):
	c = df["What country do you live in?"]
	d = df["How many years of data visualization experience do you have doing professional data visualization?"]
	x = d.replace(["less than 1", "(16-20]", "(5-10]", "(11-15]", "(21-25]", "(26-30]"], [0.5, random.randint(16, 20), random.randint(5, 10), random.randint(11, 15), random.randint(21, 25), random.randint(26, 30)]).astype(int)
	countries = ["United States", "Singapore", "United Kingdom", "Australia", "India"]
	top_five = []
	v = DataFrame({'country': c,'year_experience': x})
	for country in countries:
		print(country)
		a = v.loc[v["country"] == country].sum()
		top_five.append(a.iloc[1])

	pyplot.title("Total number of people who have experience in data visualization per country")
	y = DataFrame({"country": countries, "people": top_five})
	sns.barplot(x="country",  y="people", data=y)
	st.pyplot()

st.subheader("Bar chart")
chart1(df)
def chart2(df):
	n = df.groupby("How did you learn data visualization?").size()
	no_untake_couse = n.iloc[0]
	no_school_and_self_taught = n.iloc[1]
	no_mostly_self_taught = n.iloc[2]
	no_from_school_and_formal_course = n.iloc[3]
	total = no_untake_couse + no_school_and_self_taught + no_mostly_self_taught + no_from_school_and_formal_course
	pyplot.title("Different ways to learn data visualization")
	pyplot.pie([no_untake_couse, no_school_and_self_taught, no_mostly_self_taught, no_from_school_and_formal_course], labels=[f"Unerolled ({no_untake_couse/total * 100:.2f}%)", f"Enrolled in school and self taught ({no_school_and_self_taught/total * 100:.2f}%)", f"Mostly self taught ({no_mostly_self_taught/total*100:.2f}%)", f"Enrolled in school with formal course ({no_from_school_and_formal_course/total*100:.2f}%)"])
	st.pyplot()

st.subheader("Pie chart")
chart2(df)

def chart3(df):
	c = df["Hours a week focused on data engineering?"].head(100)
	d = df["Hours a week focused on data science?"].head(100)
	e = df["Hours a week focused on data prep work?"].head(100)
	f = df["Hours a week focused on design?"].head(100)
	pyplot.plot(c)
	pyplot.plot(d)
	pyplot.plot(e)
	pyplot.plot(f)
	pyplot.legend(["Hours per week focused on data engineering", "Hours a week focused on data science", "Hours a week focused on data prep work", "Hours a week focused on design"], loc="upper center", bbox_to_anchor=(0.5, -0.1))
	st.pyplot()

st.subheader("Line chart")
chart3(df)