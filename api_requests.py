import requests
import json
import csv
import datetime
from dotenv import load_dotenv
import os

load_dotenv()

def get_current_aq():

  api_key = os.getenv("AQ_API")

  #coordinates for Manila
  lat = "14.6042"
  lon = "120.9822"
  url = "http://api.openweathermap.org/data/2.5/air_pollution?lat="+lat+"&lon="+lon+"&appid="+api_key

  response = requests.get(url)
  data = json.loads(response.text)
  #print(data['list'][0]['components'])
  aq_list = [
    data['list'][0]['components']['co'], 
    data['list'][0]['components']['no2'],
    data['list'][0]['components']['o3'],
    data['list'][0]['components']['so2'],
    data['list'][0]['components']['pm10'],
    data['list'][0]['components']['pm2_5']
    ]
  return aq_list

def get_current_weather():
  key = os.getenv("WEATHER_API")
  location = "Manila"
  url = "http://api.weatherapi.com/v1/current.json?key="+ key + "&q=" + location

  response = requests.get(url)
  data = json.loads(response.text)
  #print(data['current'])
  weather_list = [
    data['current']['temp_c'],
    data['current']['feelslike_c'],
    data['current']['humidity'],
    data['current']['precip_mm'],
    data['current']['wind_kph'],
    ]
  return weather_list

air_quality = get_current_aq()
weather = get_current_weather()
realtime_date = datetime.datetime.now().date()
data = air_quality + weather
data.insert(0, realtime_date)

header = ["date", "co", "no2", "o3", "so2", "pm10", "pm2.5", "temp", "feelslike", "humidity", "precip", "windspeed"]

with open('datasets/current_aq_and_mf.csv', 'w', newline='') as f:
    writer = csv.writer(f)

    # write the header
    writer.writerow(header)

    # write the data
    writer.writerow(data)