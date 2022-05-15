import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
import math


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

def evaluate(model, test_features, test_labels):
    predictions = model.predict(test_features)
    errors = abs(predictions - test_labels)
    mape = 100 * np.mean(errors / test_labels)
    accuracy = 100 - mape
    print('Model Performance')
    print('Average Error: {:0.4f} degrees.'.format(np.mean(errors)))
    print('Accuracy = {:0.2f}%.'.format(accuracy))
    
    return accuracy


def plot_data(x_data,y_data1,y_data2, title):

    plt.plot(x_data, y_data1, marker =".", color = "black") #actual
    plt.plot(x_data, y_data2, marker =".", color = "#4285f4") #prediction
    plt.title(title)
    plt.xlabel("X axis") ; plt.ylabel("Y axis")
    plt.legend(['actual', 'predicted'])
    return plt.show()

def rf_model(x_train,y_train,x_test,y_test):

    n_estimators = [int(x) for x in np.linspace(start = 100, stop = 2000, num = 20)]
    max_features = ['auto', 'sqrt', 'log2']
    max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
    max_depth.append(None)
    min_samples_split = [2, 5, 10]
    min_samples_leaf = [1, 2, 4]
    bootstrap = [True, False]
    random_grid = {'n_estimators': n_estimators,
                'max_features': max_features,
                'max_depth': max_depth,
                'min_samples_split': min_samples_split,
                'min_samples_leaf': min_samples_leaf,
                'bootstrap': bootstrap}

    base_model = RandomForestRegressor(n_estimators=100, random_state=0)
    base_model.fit(x_train, np.ravel(y_train))
    base_accuracy = evaluate(base_model, x_test, y_test)
    
    model = RandomForestRegressor()
    rf_random = RandomizedSearchCV(estimator = model, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)

    rf_random.fit(x_train, np.ravel(y_train))
    print(rf_random.best_params_)
    random_accuracy = evaluate(rf_random.best_estimator_, x_test, y_test)
    y_pred = rf_random.best_estimator_.predict(x_test)

    print("Base Accuracy: ", base_accuracy)
    print("Random Accuracy: ", random_accuracy)
    print('Improvement of {:0.2f}%.'.format( 100 * (random_accuracy - base_accuracy) / base_accuracy))

    #print("actual", "predicted")
    #for i in range(len(y_test)):
        #print(y_test[i],y_pred[i])
    #print("\n")
    
    return y_test,y_pred




    



                     
