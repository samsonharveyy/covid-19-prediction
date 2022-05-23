import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from vega_datasets import data
from main import main_func
import datetime



st.title('COVID-19 Cases Prediction')
col_a, col_b = st.columns(2)
col_a.metric("Projected number of cases", 98)
col_b.subheader("Manila, Philippines")

@st.cache
def load_data():
  data = pd.read_csv('datasets/centralized_database_new.csv')
  lowercase = lambda x: str(x).lower()
  data.rename(lowercase, axis='columns', inplace=True)
  data['date'] = pd.to_datetime(data['date'])
  return data

data_load_state = st.text("Fetching data...")
data = load_data()
data_load_state.text("Showing data:") 

#if st.checkbox("Show raw data"):
#  st.subheader("Raw data")
#  st.write(data)

#actual, predicted = main_func()


np.random.seed(42)
source = pd.DataFrame(np.cumsum(np.random.randn(100, 2), 0).round(2),
                    columns=['actual', 'predicted'], index=pd.RangeIndex(100, name='date'))
source = source.reset_index().melt('date', var_name='category', value_name='cases')

# Create a selection that chooses the nearest point & selects based on x-value
nearest = alt.selection(type='single', nearest=True, on='mouseover',
                        fields=['date'], empty='none')

# The basic line
line = alt.Chart(source).mark_line(interpolate='basis').encode(
    x='date:Q',
    y='cases:Q',
    color='category:N'
)

# Transparent selectors across the chart. This is what tells us
# the x-value of the cursor
selectors = alt.Chart(source).mark_point().encode(
    x='date:Q',
    opacity=alt.value(0),
).add_selection(
    nearest
)

# Draw points on the line, and highlight based on selection
points = line.mark_point().encode(
    opacity=alt.condition(nearest, alt.value(1), alt.value(0))
)

# Draw text labels near the points, and highlight based on selection
text = line.mark_text(align='left', dx=5, dy=-5).encode(
    text=alt.condition(nearest, 'cases:Q', alt.value(' '))
)

# Draw a rule at the location of the selection
rules = alt.Chart(source).mark_rule(color='gray').encode(
    x='date:Q',
).transform_filter(
    nearest
)

# Put the five layers into a chart and bind the data
alt.layer(
    line, selectors, points, rules, text
).properties(
    width=600, height=300
)

st.altair_chart(alt.layer(line, selectors, points, rules, text), use_container_width=True)

st.caption("sampel sample sample")

col_1, col_2 = st.columns([1, 1])
col_1.metric("AQI", 52)
col_2.date_input("Date", datetime.date(2021, 1, 1))


st.subheader("Weather")
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("temperature", "70 °F", "1.2 °F")
col2.metric("feelslike", "75 °F", "1.4 °F")
col3.metric("windspeed", "9 mph", "-8%")
col4.metric("humidity", "86%", "4%")
col5.metric("precipitation", "0.0 mm", "0.0 mm")

st.subheader("Air Quality")
col6, col7, col8 = st.columns(3)
col6.metric("co", "10 ug/m3", "-1.9 ug/m3")
col7.metric("no2", "25 ug/m3", "-3.4 ug/m3")
col8.metric("o3", "9 ug/m3", "8 ug/m3")
col9, col10, col11 = st.columns(3)
col9.metric("so2", "36 ug/m3", "-4 ug/m3")
col10.metric("pm2.5", "1.5 ug/m3", "2.0 ug/m3")
col11.metric("pm10", "0.7 ug/m3", "-1.2 ug/m3")



