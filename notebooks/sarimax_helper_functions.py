import itertools
import math
import pandas as pd
import numpy as np
import warnings
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.statespace.sarimax import SARIMAX

def make_arimax_model_with_params(param_dict, time_series, train_fraction, exog_var):
    '''
    This function takes a dictionary of parameters and parameter values and makes a SARIMA model with those parameters.
    It then splits a time series into a train section and a test section. It fits the model on the train time series, then returns the fitted model, the train time series, and the test time series.
    
    It expects a parameter dictionary of the following format:
    param_dict = {'order': (p, d, q), 'seasonal': (p, d, q, m), 'trend': 'c' or 't' or 'n' or 'ct', 'enforce_stationarity': True or False, 'enforce_invertibility': True or False }
    
    It returns the fitted model (result), time_series_train, and time_series_test.
    
    If there are exogenous variables, it will return the fitted model, time_series_train, time_series_test, exog_series_train, and exog_series_test.
    '''
    warnings.filterwarnings("ignore")
    
    order_params = param_dict['order']
    seasonal_params = param_dict['seasonal']
    trend_param = param_dict['trend']
    stationarity_param = param_dict['enforce_stationarity']
    invertibility_param = param_dict['enforce_invertibility']
    
    time_series_length = len(time_series)
    time_series_train = time_series[0:math.ceil(train_fraction*time_series_length)]
    time_series_test = time_series[math.ceil(train_fraction*time_series_length):]
    
    if exog_var is None:
        exog_series_train = None
        exog_series_test = None
    else:
        exog_series_train = exog_var[0:math.ceil(train_fraction*time_series_length)]
        exog_series_test = exog_var[math.ceil(train_fraction*time_series_length):]
    
    model = SARIMAX(time_series_train, exog=exog_series_train, order=order_params, seasonal_order=seasonal_params, trend=trend_param, enforce_stationarity=stationarity_param, enforce_invertibility=invertibility_param)
    result = model.fit(display=0)
    
    if exog_var is None:
        return result, time_series_train, time_series_test, None, None
    else:
        return result, time_series_train, time_series_test, exog_series_train, exog_series_test
    

def evaluate_arimax_model_using_prediction(param_dict, time_series, train_fraction = 0.66, exog_var=None):
    '''
    Takes a parameter dictionary with keys 'order' = (p,d,q), 'seasonal' = (p,d,q,m), 'trend' = 'c' or 'ct' or 't' or 'n', 'enforce_stationarity' = True or False, 'enforce_invertibility' = True or False. Then takes a time series and a train_fraction. Uses helper function make_arimax_model_with_params to split time series into train and test sections, and fits the model on the train section.
    
    This function returns the AIC of the fitted model, and the RMSE of the model's prediction for the test section vs actual test section.
    '''
    
    result, time_series_train, time_series_test, exog_series_train, exog_series_test = make_arimax_model_with_params(param_dict, time_series, train_fraction, exog_var)
    
    test_start_date = min(time_series_test.index)
    test_end_date = max(time_series_test.index)
    
    if exog_var is None:
        test_prediction = result.predict(start=test_start_date, end=test_end_date)
    else:
        test_prediction = result.predict(start=test_start_date, end=test_end_date, exog=exog_series_test)
                                         
    test_rmse = np.sqrt(mean_squared_error(time_series_test, test_prediction))
    
    return (result.aic, test_rmse)

def get_test_prediction_from_arimax_with_params(param_dict, time_series, train_fraction = 0.66, exog_var=None):
    '''
    Takes a parameter dictionary with keys 'order' = (p,d,q), 'seasonal' = (p,d,q,m), 'trend' = 'c' or 'ct' or 't' or 'n', 'enforce_stationarity' = True or False, 'enforce_invertibility' = True or False. Then takes a time series and a train_fraction. Uses helper function make_arimax_model_with_params to split time series into train and test sections, and fits the model on the train section.
    
    This function returns the model's prediction for the test section. Prediction means that the model predicts one day ahead at each timestep.
    '''
    
    result, time_series_train, time_series_test, exog_series_train, exog_series_test = make_arimax_model_with_params(param_dict, time_series, train_fraction, exog_var)
    
    test_start_date = min(time_series_test.index)
    test_end_date = max(time_series_test.index)
    
#     test_prediction = result.predict(start=test_start_date, end=test_end_date)
    if exog_var is None:
        test_prediction = result.predict(start=test_start_date, end=test_end_date)
    else:
        test_prediction = result.predict(start=test_start_date, end=test_end_date, exog=exog_series_test)
    
    return test_prediction

def evaluate_arimax_model_using_forecasting(param_dict, time_series, train_fraction = 0.66, exog_var=None):
    '''
    Takes a parameter dictionary with keys 'order' = (p,d,q), 'seasonal' = (p,d,q,m), 'trend' = 'c' or 'ct' or 't' or 'n', 'enforce_stationarity' = True or False, 'enforce_invertibility' = True or False. Then takes a time series and a train_fraction. Uses helper function make_arimax_model_with_params to split time series into train and test sections, and fits the model on the train section.
    
    This function returns the AIC of the fitted model, and the RMSE of the model's forecast for the test section vs actual test section.
    '''
    result, time_series_train, time_series_test, exog_series_train, exog_series_test = make_arimax_model_with_params(param_dict, time_series, train_fraction, exog_var)
    
    if exog_var is None:
        test_forecast = result.forecast(steps = len(time_series_test))
    else:
        test_forecast = result.forecast(steps = len(time_series_test), exog=exog_series_test)
    
    test_rmse = np.sqrt(mean_squared_error(time_series_test, test_forecast))
    
    return (result.aic, test_rmse)

def get_test_forecast_from_arimax_with_params(param_dict, time_series, train_fraction = 0.66, exog_var=None):
    '''
    Takes a parameter dictionary with keys 'order' = (p,d,q), 'seasonal' = (p,d,q,m), 'trend' = 'c' or 'ct' or 't' or 'n', 'enforce_stationarity' = True or False, 'enforce_invertibility' = True or False. Then takes a time series and a train_fraction. Uses helper function make_arimax_model_with_params to split time series into train and test sections, and fits the model on the train section.
    
    This function returns the model's forecast for the test section. Prediction means that the model forecasts for the entire length of the time series, using its own previous predictions to continue forecasting ahead.
    '''
    result, time_series_train, time_series_test, exog_series_train, exog_series_test = make_arimax_model_with_params(param_dict, time_series, train_fraction, exog_var)
#     test_forecast = result.forecast(steps = len(time_series_test))

    if exog_var is None:
        test_forecast = result.forecast(steps = len(time_series_test))
    else:
        test_forecast = result.forecast(steps = len(time_series_test), exog=exog_series_test)
    
    return test_forecast

def make_all_possible_param_dictionaries(param_choices):
    '''
    Takes a dictionary with parameters as keys, and a list of possible parameter values as the value for each key.
    Returns a list of dictionaries with all possible combinations of parameter values.
    
    Requires itertools.product to work.
    
    Example:
    param_choices = {'order': ['linear', 'quadratic', 'cubic'], 'intercept': [0, 10]}
    
    make_all_possible_param_dictionaries(param_choices) outputs:
    [ {'order': 'linear', 'intercept': 0}, {'order': 'quadratic', 'intercept': 0}, {'order': 'cubic', intercept: '0'},
      {'order': 'linear', 'intercept': 10}, {'order': 'quadratic', 'intercept': 10}, {'order': 'cubic', intercept: 10}
    ]
    '''
    
    master_list_of_dictionaries = []
    
    param_combinations = list(itertools.product(*param_choices.values()))
    parameter_names = list(param_choices.keys())
    
    for combo in param_combinations:
        temp_dict = dict(zip(parameter_names, combo))
        master_list_of_dictionaries.append(temp_dict)
        
    return master_list_of_dictionaries

def max_variable_in_exog_df(var, exog_df, train_fraction):
    
    max_value_for_var = {
        'obesity_prevalence': 1,
        'diabetes_prevalence': 1,
        'high_cholesterol': 1,
        'exercise': 1,
        'hypertension': 1,
        'general_health': 1,
        'mental_health': 30,
        'coverage': 1,
        'income': 5,
        'smoker': 1,
        'med_cost': 1,
        'gender': 1
    }
    
    time_series_length = exog_df.shape[0]
    
    exog_series_train = exog_df[0:math.ceil(train_fraction*time_series_length)]
    exog_series_test = exog_df[math.ceil(train_fraction*time_series_length):]
    
    exog_series_test[var] = max_value_for_var[var]
    
    exog_maxed_var_df = pd.concat([exog_series_train, exog_series_test])
    
    return exog_maxed_var_df