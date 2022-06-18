import os
import math
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

def feature_drop(feature_data):
    cor_matrix = feature_data.corr().abs()
    upper_tri = cor_matrix.where(np.triu(np.ones(cor_matrix.shape),k=1).astype(bool))
    to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > 0.5)]
    
    print(cor_matrix)
    print(to_drop)
    
    sns.heatmap(cor_matrix)
    plt.show()

def preprocess_dataset(x_data,y_data):
    x_train,x_test,y_train,y_test = train_test_split(x_data,y_data,test_size=0.2,random_state=0)
    scale = StandardScaler()
    x_train = scale.fit_transform(x_train); x_test = scale.fit_transform(x_test)
    y_train = scale.fit_transform(y_train); y_test = scale.fit_transform(y_test)
    
    y_pred = polynomial_regression(x_train,x_test,y_train,y_test)
    
    y_test = scale.inverse_transform(y_test)
    y_pred = scale.inverse_transform(y_pred)
    
    plot(y_test,y_pred)
    performance_metric(x_data,y_test,y_pred)

def curve_visualization(x_data,y_data,param,degree,title):
    x_train,x_test,y_train,y_test = train_test_split(x_data,y_data,test_size=0.2,random_state=0)
    scale = StandardScaler()
    x_train = scale.fit_transform(x_train)
    y_train = scale.fit_transform(y_train)

    poly = PolynomialFeatures(degree,include_bias=False)
    x_poly_train = poly.fit_transform(x_train)
    poly.fit(x_train,y_train)
    
    model = LinearRegression().fit(x_poly_train,y_train)
    x_grid_train = np.arange(min(x_train), max(x_train) + .01, step=0.01)
    x_grid_train = x_grid_train.reshape(len(x_grid_train), 1)
    
    y_pred = model.predict(poly.fit_transform(x_grid_train))
    
    fig_name = os.path.join(CURRENT_DIR,title)
    plt.scatter(x_train, y_train, color='red',  marker =".",label='Training Data Points')
    plt.plot(x_grid_train,y_pred, color='blue', label='Model Curve')
    plt.title(title)
    plt.legend([param,'curve'])
    
    plt.savefig(fig_name)

    plt.show()
    
def polynomial_regression(x_train,x_test,y_train,y_test):
    poly = PolynomialFeatures(degree, include_bias=False)

    x_poly = poly.fit_transform(x_train)
    poly.fit(x_train,y_train)
    model = LinearRegression().fit(x_poly,y_train)

    y_pred = model.predict(poly.fit_transform(x_test))
    return y_pred
    
def plot(y_test,y_pred):
    title = 'polynomial regression - Degree '+str(degree)
    fig_name = os.path.join(CURRENT_DIR,title)

    plt.plot(range(len(y_test)),y_test, color='blue')
    plt.plot(range(len(y_pred)),y_pred, color='red')

    plt.title(title)
    plt.legend(['actual','predicted'])
    plt.xlabel("X axis") ; plt.ylabel("Y axis")
    plt.xticks(rotation=-90,fontsize=5)
    
    #plt.savefig(fig_name)
    plt.show()

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
    
        print("MSE    : ", float("{0:.5f}".format(MSE_val)))
        print("RMSE   : ", float("{0:.5f}".format(RMSE_val)))
        print("MAPE   : ", float("{0:.5f}".format(MAPE_val)))
        print("MAE    :", float("{0:.5f}".format(MAE_val)))
        print("R2     :", float("{0:.5f}".format(R2_val)))
        print("ADJ R2 :", float("{0:.5f}".format(ADJR2_val)))
        print("\n")

def remove_outlier_IQR(df):
    Q1=df.quantile(0.25)
    Q3=df.quantile(0.75)
    IQR=Q3-Q1
    df_final = df[~((df < (Q1 - 1.5 * IQR)) |(df > (Q3 + 1.5 * IQR))).any(axis=1)]
    return df_final

def remove_outlier_ZScore(df):
    from scipy import stats
    df_final = df[(np.abs(stats.zscore(df)) < 3).all(axis=1)]
    return df_final


# ----------------------------------------------------------------------------------------------

degree = 2
CURRENT_DIR = os.path.dirname(__file__)

def preprocess_streamlit(x_data, y_data):
    x_train,x_test,y_train,y_test = train_test_split(x_data,y_data,test_size=0.2,random_state=0)
    scale = StandardScaler()
    x_train = scale.fit_transform(x_train); x_test = scale.fit_transform(x_test)
    y_train = scale.fit_transform(y_train); y_test = scale.fit_transform(y_test)

    #change testing set to realtime data
    data_2 = pd.read_csv('datasets/current_aq_and_mf.csv')
    realtime_data = data_2[["pm2.5","o3","temp","humidity","precip","windspeed"]]
    realtime_data = scale.fit_transform(realtime_data)
    realtime_data = scale.inverse_transform(realtime_data)
    y_test_sample = [[200]] #placeholder
    y_test_sample = scale.fit_transform(y_test_sample)
    y_test_sample = scale.inverse_transform(y_test_sample)

    y_pred = polynomial_regression(x_train,realtime_data,y_train,y_test_sample)
    
    y_test_sample = scale.inverse_transform(y_test)
    y_pred = scale.inverse_transform(y_pred)
    
    plot(y_test_sample, y_pred)
    #performance_metric(x_data,y_test,y_pred)
    return y_test_sample, y_pred

def streamlit_polynomial_reg():
    input_path = os.path.join(CURRENT_DIR,"datasets/centralized_database_new.csv")
    data = pd.read_csv(input_path)
    data = data[["pm2.5","o3","temp","humidity","precip","windspeed","cases"]]
    x_data = data[["pm2.5","o3","temp","humidity","precip","windspeed"]]
    y_data = data[["cases"]]

    x_data = x_data.to_numpy()
    y_data = y_data.to_numpy()


    actual, predicted = preprocess_streamlit(x_data,y_data)
    return actual, predicted

def main():
    input_path = os.path.join(CURRENT_DIR,"datasets/centralized_database_new.csv")
    data = pd.read_csv(input_path)

    #data = data[["pm2.5","pm10","co","so2","no2","o3","temp","feelslike","humidity","precip","windspeed","cases"]]
    data = data[["pm2.5","o3","temp","humidity","precip","windspeed","cases"]]
    #data = remove_outlier_IQR(data)

    x_data = data[["pm2.5","o3","temp","humidity","precip","windspeed"]]
    y_data = data[["cases"]]

    #feature_drop(data)

    x_data = x_data.to_numpy()
    y_data = y_data.to_numpy()

    preprocess_dataset(x_data,y_data)

if __name__ == "__main__":
    main()
