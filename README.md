# COVID-19 Prediction Dashboard in Streamlit
Capstone requirement for CoE199 (de Padua, Lagonsin, Samson)

A visualization of the prediction of the machine learning models on COVID-19 daily incidence using air quality and meteorological factors as input features.

Linear regression, polynomial regression, and random forest regression are the machine learning models used.

## Installation requirements
```
pip3 install -r requirements.txt
```

## Prerequisites
The model uses APIs from OpenWeather and WeatherAPI to get daily air quality and weather data for daily projection of COVID-19 cases. This is primarily done 
for visualization purposes since there is delayed reporting in the country's daily COVID cases. Get active API keys to and pass it as variables before running the
file: 
```
python3 api_requests.py
```
Running the command should replace the values in current_aq_and_mf.csv with new daily readings.

## Running the App
The application is built using Streamlit and uses the three machine learning model Python files to generate predictions. Make sure to have streamlit installed and to run:
```
streamlit run streamlit_app.py
```

Random forest regression is the recommended model since it displayed highest values of R2 and Adjusted R2 at 0.29 and 0.17
(although not extremely good), followed by polynomial regression at 0.13 and 0.9, and linear regression at 0.08 and 0.06. 

## References
* [NumPy](https://numpy.org/doc/stable/)
* [OpenWeather Air Pollution API](https://openweathermap.org/api/air-pollution)
* [Pandas](https://pandas.pydata.org/docs/)
* [Scikit-Learn](https://scikit-learn.org/stable/modules/classes.html)
* [Streamlit](https://docs.streamlit.io/library/api-reference)
* [WeatherAPI](https://www.weatherapi.com/docs/)





