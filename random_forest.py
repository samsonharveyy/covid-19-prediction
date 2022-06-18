import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
import math
from sklearn.preprocessing import StandardScaler
import pandas as pd
import streamlit


def split_data(x_data, y_data):
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.15, random_state=0)
    return x_train, x_test, y_train, y_test

def evaluate(model, test_features, test_labels):
    predictions = model.predict(test_features)
    errors = abs(predictions - test_labels)
    mape = 100 * np.mean(errors / test_labels)
    accuracy = 100 - mape
    print('Average Error: {:0.4f} degrees.'.format(np.mean(errors)))
    print('Accuracy = {:0.2f}%.'.format(accuracy))
    return accuracy

def randomized_search_grid():
    n_estimators = [int(x) for x in np.linspace(start = 100, stop = 2000, num = 20)]
    max_features = ['auto', 'sqrt', 'log2']
    max_depth = [int(x) for x in np.linspace(10, 200, num = 20)]
    max_depth.append(None)
    min_samples_split = [2, 3, 5, 8, 10]
    min_samples_leaf = [1, 2, 3, 4, 5, 8]
    bootstrap = [True, False]
    random_grid = {'n_estimators': n_estimators,
                'max_features': max_features,
                'max_depth': max_depth,
                'min_samples_split': min_samples_split,
                'min_samples_leaf': min_samples_leaf,
                'bootstrap': bootstrap}
    return random_grid

def params_grid_search():
    #based from the best params output of randomized search CV
    param_grid = {
        'bootstrap': [False, True],
        'max_depth': [50, 70, 100],
        'max_features': ['sqrt', 'log2'],
        'min_samples_leaf': [2, 3, 5],
        'min_samples_split': [2, 3, 5],
        'n_estimators': [100, 1000, 1500]
    }
    return param_grid

def randomized_search_CV(x_train, y_train, model):
    random_grid = randomized_search_grid()
    rf_random = RandomizedSearchCV(estimator=model, param_distributions=random_grid, n_iter=100, cv=10, verbose=2, 
                                   random_state=0, n_jobs = -1)

    rf_random.fit(x_train, np.ravel(y_train))
    return rf_random

def grid_search_CV(x_train, y_train, model):
    param_grid = params_grid_search()
    rf_grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=10, verbose=2, n_jobs = -1)
    rf_grid.fit(x_train, np.ravel(y_train))
    return rf_grid

def rf_model(x_train,y_train,x_test,y_test):
    #best params from Grid and Randomized Search CV
    model = RandomForestRegressor(bootstrap=True, max_depth=70, max_features='log2', 
                                min_samples_leaf=2, min_samples_split=3, n_estimators=1500, random_state=0,
                                )
    model.fit(x_train, np.ravel(y_train))
    y_pred = model.predict(x_test)

    #print("actual", "predicted")
    #for i in range(len(y_test)):
        #print(y_test[i],y_pred[i])
    #print("\n")
    
    return y_test, y_pred

def plot_data(x_data, y_data1, y_data2, title):

    plt.plot(x_data, y_data1, marker =".", color = "#2f7fe1") #actual
    plt.plot(x_data, y_data2, marker =".", color = "#4b00bf") #predicted
    plt.title(title)
    plt.xlabel("X axis") ; plt.ylabel("Y axis")
    plt.legend(['actual', 'predicted'])
    return plt.show()

def performance_metric(y_test,y_pred, x_data):
        from sklearn.metrics import mean_squared_error
        MSE_val = mean_squared_error(y_test, y_pred)
        RMSE_val = math.sqrt(MSE_val)

        from sklearn.metrics import mean_absolute_percentage_error
        MAPE_val = mean_absolute_percentage_error(y_test, y_pred)

        from sklearn.metrics import mean_absolute_error
        MAE_val = mean_absolute_error(y_test,y_pred)

        from sklearn.metrics import r2_score
        R2_val = r2_score(y_test,y_pred)
        ADJR2_val = 1 - (1-R2_val) * (len(y_test)-1)/(len(y_test)-x_data.shape[1]-1)

        print("MSE    : ", MSE_val)
        print("RMSE   : ", RMSE_val)
        print("MAPE   : ", MAPE_val)
        print("MAE    :", MAE_val)
        print("R2     :", R2_val)
        print("ADJ R2 :", ADJR2_val)
        print("\n")

def streamlit_random_forest():
  #read database
  data = pd.read_csv('datasets/centralized_database_new.csv')

  #get input and output features
  features = ["temp", "feelslike", "humidity", "windspeed", "precip", "pm2.5", "pm10","co","so2","no2","o3"]
  x_data = data[features]
  y_data = data.iloc[:,12:].values

  #reshape data
  scaler = StandardScaler()

  x_data = scaler.fit_transform(x_data)
  y_data = scaler.fit_transform(y_data)

  x_data = scaler.inverse_transform(x_data)
  y_data = scaler.inverse_transform(y_data)

  #build and run model
  rf_x_train, rf_x_test, rf_y_train, rf_y_test = split_data(x_data, y_data)

  title = "Random Forest Plot"

  #change testing set to realtime data
  data_2 = pd.read_csv('datasets/current_aq_and_mf.csv')
  realtime_data = data_2[features]
  realtime_data = scaler.fit_transform(realtime_data)
  realtime_data = scaler.inverse_transform(realtime_data)
  y_test = [[200]] #placeholder

  prediction = rf_model(rf_x_train, rf_y_train, realtime_data, y_test)
  #plot_data(range(len(prediction[0])), prediction[0], prediction[1], title)
  #performance_metric(prediction[0], prediction[1], x_data)
  return prediction[0], prediction[1]

def main():
  #read database
  data = pd.read_csv('datasets/centralized_database_new.csv')

  #get input and output features
  features = ["temp", "feelslike", "humidity", "windspeed", "precip", "pm2.5", "pm10","co","so2","no2","o3"]
  x_data = data[features]
  y_data = data.iloc[:,12:].values

  #reshape data
  scaler = StandardScaler()
  x_data = scaler.fit_transform(x_data)
  y_data = scaler.fit_transform(y_data)

  x_data = scaler.inverse_transform(x_data)
  y_data = scaler.inverse_transform(y_data)


  #build and run model
  rf_x_train, rf_x_test, rf_y_train, rf_y_test = split_data(x_data, y_data)

  title = "Random Forest Plot"

  prediction = rf_model(rf_x_train, rf_y_train, rf_x_test, rf_y_test)
  plot_data(range(len(prediction[0])), prediction[0], prediction[1], title)
  performance_metric(prediction[0], prediction[1], x_data)


if __name__ == "__main__":
  main()
                
