import streamlit as st
import pandas as pd
import numpy as np




# Title
st.title("Uber pickups in NYC")

#FETCH DATA

DATE_COLUMN = 'date/time'
DATA_URL = ('https://s3-us-west-2.amazonaws.com/''streamlit-demo-data/uber-raw-data-sep14.csv.gz')

@st.cache
def load_data(nrows):
	df = pd.read_csv(DATA_URL, nrows=nrows)
	lowercase = lambda x: str(x).lower()
	df.rename(lowercase, axis='columns', inplace=True)
	df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN])
	return df


# Create a text element and let the reader know the data is loading.
data_load_state = st.text('Loading data...')
df = load_data(1000)
# Notify the reader that the data was successfully loaded.
data_load_state.text("Done! (using st.cache)")
if st.checkbox('Show raw data'):
    st.subheader('Raw data')
    st.write(df)

# Histogram
st.subheader('Number of pickups by hour')
hist_values = np.histogram(df[DATE_COLUMN].dt.hour, bins=24, range=(0,24))[0]
st.bar_chart(hist_values)

#Data Map
st.subheader('Map of all pickups')
st.map(df)

hour_to_filter = st.slider('hour', 0, 23, 17)  # min: 0h, max: 23h, default: 17h
filtered_data = df[df[DATE_COLUMN].dt.hour == hour_to_filter]
st.subheader(f'Map of all pickups at {hour_to_filter}:00')
st.map(filtered_data)