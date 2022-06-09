import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from vega_datasets import data
from main import main_func
import datetime
import time

@st.cache
def load_data():
  data = pd.read_csv('datasets/centralized_database_new.csv')
  lowercase = lambda x: str(x).lower()
  data.rename(lowercase, axis='columns', inplace=True)
  data['date'] = pd.to_datetime(data['date'])
  return data

st.title('COVID-19 Cases Prediction')
c1, c2 = st.columns(2)
aqi = c1.metric("AQI", 52)
date = c2.date_input("Date", datetime.date(2021, 1, 1), min_value=datetime.date(2020, 11, 25), max_value=datetime.date(2022, 5, 7))

c3, c4 = st.columns(2)
cases = c3.metric("Projected number of cases", 98)
area = c4.subheader("Manila, Philippines")

data_load_state = st.text("Fetching data...")
data = load_data()
data_load_state.text("Showing data:") 

if st.checkbox("Show raw data"):
  st.subheader("Raw data")
  st.write(data)

indexes = data.index
index_finder = data.index[data["date"]==pd.to_datetime(date)].tolist()
index = index_finder[0]


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


#weather variables
if index == 0:
    temp_prev = 0
    feelslike_prev = 0
    windspeed_prev = 0
    humidity_prev = 0
    precip_prev = 0
else:
    temp_prev = round(data["temp"][index] - data["temp"][index-1], 2)
    feelslike_prev = round(data["feelslike"][index] - data["feelslike"][index-1], 2)
    windspeed_prev = round(data["windspeed"][index] - data["windspeed"][index-1], 2)
    humidity_prev = round(data["humidity"][index] - data["humidity"][index-1], 2)
    precip_prev = round(data["precip"][index] - data["precip"][index-1], 2)

temp = data["temp"][index]
feelslike = data["feelslike"][index]
windspeed = data["windspeed"][index]
humidity = data["humidity"][index]
precip = data["precip"][index]


st.subheader("Weather")
col1, col2, col3 = st.columns(3)
col1.metric("temperature", str(temp)+" °C", str(temp_prev)+" °C")
col2.metric("feelslike", str(feelslike) + " °C", str(feelslike_prev) + " °C")
col3.metric("windspeed", str(windspeed) + " km/hr", str(windspeed_prev) + " km/hr")
col4, col5 = st.columns(2)
col4.metric("humidity", humidity, humidity_prev)
col5.metric("precipitation", str(precip) + " mm", str(precip_prev) + " mm")

#air quality variables
if index == 0:
    co_prev = 0
    no2_prev = 0
    o3_prev = 0
    so2_prev = 0
    pm25_prev = 0
    pm10_prev = 0
else:
    co_prev = round(data["co"][index] - data["co"][index-1], 2)
    no2_prev = round(data["no2"][index] - data["no2"][index-1], 2)
    o3_prev = round(data["o3"][index] - data["o3"][index-1], 2)
    so2_prev = round(data["so2"][index] - data["so2"][index-1], 2)
    pm25_prev = round(data["pm2.5"][index] - data["pm2.5"][index-1], 2)
    pm10_prev = round(data["pm10"][index] - data["pm10"][index-1], 2)

co = round(data["co"][index], 2)
no2 = round(data["no2"][index], 2)
o3 = round(data["o3"][index], 2)
so2 = round(data["so2"][index], 2)
pm25 = round(data["pm2.5"][index], 2)
pm10 = round(data["pm10"][index], 2)

st.subheader("Air Quality")
col6, col7, col8 = st.columns(3)
col6.metric("co", str(co) + " ug/m3", str(co_prev) + " ug/m3")
col7.metric("no2", str(no2) + " ug/m3", str(no2_prev) + " ug/m3")
col8.metric("o3", str(o3) + " ug/m3", str(o3_prev) + " ug/m3")
col9, col10, col11 = st.columns(3)
col9.metric("so2", str(so2) + " ug/m3", str(so2_prev) + " ug/m3")
col10.metric("pm2.5", str(pm25) + " μg/m3", str(pm25_prev) + " μg/m3")
col11.metric("pm10", str(pm10) + " μg/m3", str(pm10_prev) + " μg/m3")



