import streamlit as st
import pandas as pd
import numpy as np




# Title
st.title("Uber pickups in NYC")

#FETCH DATA

DATE_COLUMN = 'date/time'
DATA_URL = ('https://s3-us-west-2.amazonaws.com/''streamlit-demo-data/uber-raw-data-sep14.csv.gz')




def load_data(nrows):
	df = pd.read_csv(DATA_URL, nrows=nrows)
	return df

# https://docs.streamlit.io/en/stable/tutorial/create_a_data_explorer_app.html