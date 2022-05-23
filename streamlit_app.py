import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

st.title('COVID-19 Case Prediction')

@st.cache
def load_data():
  data = pd.read_csv('datasets/centralized_database_new.csv')
  lowercase = lambda x: str(x).lower()
  data.rename(lowercase, axis='columns', inplace=True)
  data['date'] = pd.to_datetime(data['date'])
  return data

data_load_state = st.text("Fetching data...")
data = load_data()
data_load_state.text("Done!") 

if st.checkbox("Show raw data"):
  st.subheader("Raw data")
  st.write(data)

