import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import seaborn as sns

def plot_data(x_data,y_data1,y_data2,title):

    plt.plot(x_data, y_data1, marker =".", color = "#2f7fe1", label='actual') #actual
    plt.plot(x_data, y_data2, marker =".", color = "#4b00bf", label='prediction') #prediction
    plt.legend(loc="upper right")
    plt.title(title)
    plt.xlabel("X axis") ; plt.ylabel("Y axis")
    plt.xticks([])
    return plt.show()

def linear_model(x_train,y_train,x_test,y_test):

    model = LinearRegression()
    model.fit(x_train,y_train)

    y_pred = model.predict(x_test)
    
    print("actual", "predicted")
    #for i in range(len(y_test)):
    #    print(y_test[i],y_pred[i])
    #print("\n")

    return y_test,y_pred

def performance_metric(x_data, y_test,y_pred):
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

def remove_outlier_IQR(df):
    Q1=df.quantile(0.25)
    Q3=df.quantile(0.75)
    IQR=Q3-Q1
    df_final=df[~((df<(Q1-1.5*IQR)) | (df>(Q3+1.5*IQR)))]
    return df_final    

def remove_outlier_ZScore(df):
    from scipy import stats
    df_final = df[(np.abs(stats.zscore(df)) < 3).all(axis=1)]
    return df_final

def logtrans(x):
    return np.log(x)

def sqrttrans(x):
    return np.sqrt(x)    

def streamlit_linear_reg():
    CURRENT_DIR = os.path.dirname(__file__)
    file_path = os.path.join(CURRENT_DIR, "datasets/centralized_database_new.csv")

    df = pd.read_csv(file_path)
    df = df[['cases', 'temp', 'o3']] # Put parameters to include here

    x_data = df[['temp', 'o3']]
    y_data = df[['cases']]

    # No Scaling
    x_data = x_data.to_numpy()
    y_data = y_data.to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=0)
    title = "[No Outliers] temp + o3 vs cases"

    #change testing set to realtime data
    data_2 = pd.read_csv('datasets/current_aq_and_mf.csv')
    realtime_data = data_2[['temp', 'o3']]
    realtime_data = realtime_data.to_numpy()
    y_test_sample = [[200]] #placeholder

    prediction = linear_model(X_train,y_train,realtime_data,y_test_sample)
    #plot_data(range(len(prediction[0])),prediction[0],prediction[1],title)
    #performance_metric(x_data,prediction[0],prediction[1])

    return prediction[0], prediction[1]

def main():
    CURRENT_DIR = os.path.dirname(__file__)
    file_path = os.path.join(CURRENT_DIR, "datasets/centralized_database_new.csv")

    df = pd.read_csv(file_path)
    df = df[['cases', 'temp', 'o3']] # Put parameters to include here

    print(df.shape)

    # Code for multivariate linear regression

    x_data = df[['temp', 'o3']]
    y_data = df[['cases']]

    # No Scaling
    x_data = x_data.to_numpy()
    y_data = y_data.to_numpy()

    # Linear Transformation
    # x_data = np.log(x_data)
    # y_data = np.log(y_data)


    X_train, X_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=0)
    title = "[No Outliers] temp + o3 vs cases"
    prediction = linear_model(X_train,y_train,X_test,y_test)
    #plot_data(range(len(prediction[0])),prediction[0],prediction[1],title)
    plot_data(range(len(prediction[0])),prediction[0],prediction[1],title)
    performance_metric(x_data,prediction[0],prediction[1])

    print(df.shape)

if __name__ == "__main__":
    main()
