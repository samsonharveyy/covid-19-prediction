from random import Random
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import RFECV
from sklearn.model_selection import train_test_split
import math


def split_data(x_data, y_data):
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=0)
    return x_train, x_test, y_train, y_test

def evaluate(model, test_features, test_labels):
    predictions = model.predict(test_features)
    errors = abs(predictions - test_labels)
    mape = 100 * np.mean(errors / test_labels)
    accuracy = 100 - mape
    print('Model Performance')
    print('Average Error: {:0.4f} degrees.'.format(np.mean(errors)))
    print('Accuracy = {:0.2f}%.'.format(accuracy))
    return accuracy

def randomized_search_grid():
    n_estimators = [int(x) for x in np.linspace(start = 100, stop = 2000, num = 20)]
    max_features = ['auto', 'sqrt', 'log2']
    max_depth = [int(x) for x in np.linspace(10, 200, num = 20)]
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
    return random_grid

def params_grid_search():
    #based from the best params output of randomized search cv
    param_grid = {
        'bootstrap': [True],
        'max_depth': [80, 90, 100],
        'max_features': [2, 3],
        'min_samples_leaf': [3, 4, 5],
        'min_samples_split': [8, 10, 12],
        'n_estimators': [100, 200, 500, 6000]
    }
    return param_grid

def feature_selection(x_train, y_train, features):
    model = RandomForestRegressor(random_state=0)
    model.fit(x_train, np.ravel(y_train))

    #calculate feature importance
    importance = model.feature_importances_
    #for i, j in enumerate(importance):
        #print('Feature: %0d, Score: %.5f' % (i,j))
    #plt.bar([x for x in range(len(importance))], importance)
    #plt.show()

    #for visualization, bar graph with feature labels
    feature_importance = list(zip(features,model.feature_importances_))
    feature_importance.sort(key = lambda x : x[1])
    #plt.barh([x[0] for x in feature_importance],[x[1] for x in feature_importance])
    #plt.show()
    
    #feature selection, to modify more
    rfecv = RFECV(estimator=model, step=1, min_features_to_select=5, cv=5, scoring="r2")
    return rfecv

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
    #base model for reference
    base_model = RandomForestRegressor(n_estimators=100, random_state=0)
    base_model.fit(x_train, np.ravel(y_train))
    #y_pred = base_model.predict(x_test)
    base_accuracy = evaluate(base_model, x_test, y_test)

    model = RandomForestRegressor()

    #Randomized Search CV
    #rf_random = randomized_search_CV(x_train, y_train, model)
    #print(rf_random.best_params_)
    #random_accuracy = evaluate(rf_random.best_estimator_, x_test, y_test)
    #y_pred = rf_random.best_estimator_.predict(x_test)

    #Grid Search CV
    #rf_grid = grid_search_CV(x_train, y_train, model)
    #print(rf_grid.best_params_)
    #y_pred = rf_grid.best_estimator_.predict(x_test)

    #best params from Grid and Randomized Search CV
    model = RandomForestRegressor(bootstrap=False, max_depth=130, max_features='sqrt', 
                                min_samples_leaf=2, min_samples_split=2, n_estimators=900, random_state=0,
                                verbose=2, n_jobs=-1)
    model.fit(x_train, np.ravel(y_train))
    y_pred = model.predict(x_test)

    #print("Base Accuracy: ", base_accuracy)
    #print("Random Accuracy: ", random_accuracy)
    #print('Improvement of {:0.2f}%.'.format( 100 * (random_accuracy - base_accuracy) / base_accuracy))

    #print("actual", "predicted")
    #for i in range(len(y_test)):
        #print(y_test[i],y_pred[i])
    #print("\n")
    
    return y_test, y_pred

def plot_data(x_data, y_data1, y_data2, title):

    plt.plot(x_data, y_data1, marker =".", color = "black") #actual
    plt.plot(x_data, y_data2, marker =".", color = "#4285f4") #prediction
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


    



                     
