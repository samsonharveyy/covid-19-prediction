from rf_helper import performance_metric, evaluate, plot_data, rf_model, feature_selection, split_data

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import TimeSeriesSplit
import pandas as pd 


#read database
data = pd.read_csv('datasets/centralized_database_old.csv')
#data = pd.read_csv('datasets/centralized_database_new.csv')

#get input and output features
#features = ["temp", "feelslike", "humidity", "windspeed", "precip", "pm2.5", "pm10","co","so2","no2","o3"]
#x_data = data[features]
x_data = data[['humidity', 'pm2.5', 'precip', 'o3', 'temp']]
#x_data = data[['temp', 'humidity', 'pm2.5', 'precip', 'o3', 'temp']]
y_data = data.iloc[:,13:].values


#transform data
scaler = StandardScaler()
x_data = scaler.fit_transform(x_data)
y_data = scaler.fit_transform(y_data)

#x_data = scaler.inverse_transform(x_data)
#y_data = scaler.inverse_transform(y_data)



#build and run model
rf_x_train, rf_x_test, rf_y_train, rf_y_test = split_data(x_data, y_data)
#feature_selection(rf_x_train, rf_y_train, features)


title = "Feature Selection - Random Forest"


#testing set
#prediction = rf_model(rf_x_train, rf_y_train, rf_x_test, rf_y_test)
#plot_data(range(len(prediction[0])), prediction[0], prediction[1], title)
#performance_metric(prediction[0], prediction[1], x_data)

#training set
prediction = rf_model(rf_x_train, rf_y_train, rf_x_train, rf_y_train)
plot_data(range(len(prediction[0])), prediction[0], prediction[1], title)
performance_metric(prediction[0], prediction[1], x_data)

