import streamlit as st
import numpy as np
import pandas as pd


# TITLE COMMANDS
# st.title("Royce")


# WRITE COMMANDS
# st.write("Here's our first attempt at using data to create a table:")
# st.write(pd.DataFrame({
#     'first column': [1, 2, 3, 4],
#     'second column': [10, 20, 30, 40]
# }))



# MAGIC COMMANDS
# Draw a title and some text to the app:
# '''
# # This is the document title

# This is some _markdown_.
# '''
# x = 1
# x

df = pd.DataFrame({
  'first column': [1, 2, 3, 4],
  'second column': [10, 20, 30, 40]
})

# Charts
# Line chart
# chart_data = pd.DataFrame(
# 	np.random.randn(20, 3),
# 	columns = ["a", "b", "c"]
# )
# chart_data
# st.line_chart(chart_data)

# Plot Map
# map_data = pd.DataFrame(
#     np.random.randn(1000, 2 )/ [50, 50] + [37.76, -122.4],
#     columns=['lat', 'lon'])

# st.map(map_data)

# Widgets
# Checkbox
# a = st.checkbox("Show dataframe")
# if a:
#     chart_data = pd.DataFrame(
#        np.random.randn(20, 3),
#        columns=['a', 'b', 'c'])

#     st.line_chart(chart_data)

# Selectbox
option = st.selectbox(
    'Which number do you like best?',
     df['first column'])

'You selected: ', option