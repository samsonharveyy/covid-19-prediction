from rf_helper import performance_metric, evaluate, plot_data, rf_model

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
import pandas as pd 
from sklearn.model_selection import train_test_split



data = pd.read_csv('datasets/centralized_database.csv')

#x_data = data[['temp', 'humidity', 'precip', 'windspeed', 'pm2.5', 'o3']]
#x_data = data[["temp", "feelslike", "humidity", "windspeed", "precip", "pm2.5", "pm10","co","so2","no2","o3"]]
x_data = data[['humidity', 'pm2.5', 'precip', 'o3', 'temp']]

y_data = data.iloc[:,13:].values


scaler = MinMaxScaler()
x_data = scaler.fit_transform(x_data)
y_data = scaler.fit_transform(y_data)

x_data = scaler.inverse_transform(x_data)
y_data = scaler.inverse_transform(y_data)


title = "[No Scale] Testing Set - Random Forest (Dropped)"
rf_x_train, rf_x_test, rf_y_train, rf_y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=0)
prediction = rf_model(rf_x_train, rf_y_train, rf_x_test, rf_y_test)
plot_data(range(len(prediction[0])), prediction[0], prediction[1], title)




performance_metric(prediction[0], prediction[1], x_data)

