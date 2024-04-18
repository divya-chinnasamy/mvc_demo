# Linear Regression Algorithm:
import os
import pickle
import pandas as pd
import json
import numpy as np
from time import sleep
import sys
import os
import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold 
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import *
from sklearn.model_selection import cross_val_score
from statistics import mean, stdev
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from math import sqrt
import logging
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

# parameters:
'''fit_intercept:bool, default=True
copy_X bool, default=True
n_jobs int, default=None
positive bool, default=False'''


def load():
    try:
        with open('conf_reg.json', 'r')as file:
            con = json.load(file)
        return 1, con
    except:
        d1 = {'Status':'0', 'Error': 'No such file or Directory', 'Error_code': 'A230'}
        return 0, d1
def lr_reg_model(con):
    try:
        df = pd.read_csv(con["lr_regression"]["input_file"])
        x = df[con["lr_regression"]["IV"]]
        y = df[con["lr_regression"]["DV"]]
        new_df = pd.concat([x, y], axis=1)
        min_required_rows = 100
        min_required_columns = 5
        if len(new_df) >= min_required_rows and len(new_df.columns) >= min_required_columns:
            x_support = x.apply(lambda row: all(isinstance(value, (int, float)) for value in row), axis=1).all()
            y_support = y.apply(lambda row: all(isinstance(value, (int, float)) for value in row), axis=1).all()
            if x_support == True and y_support == True:
                check_null1 = x.isna().any().any()
                check_null2 = y.isna().any().any()
                if check_null1 == False and check_null2 == False:
                    correlations = x.corrwith(y)
                    No_cor = []
                    # checking correlation in independent and Dependent Features:
                    for i in correlations.index:
                        res = 0.1 <= correlations[i] <= -0.1
                        if res == True:
                            No_cor.append(i)
                        else:
                            pass
                    if len(No_cor) != 0:
                        d1 = {'Status': '0', 'Correl_Interpretation':No_cor, 'Error': 'Poorly Correlated', 'Error_code': 'A235'}
                        return d1
                    else:
                        X = np.asarray(x)
                        Y = np.asarray(y)
                        fit=con["lr_regression"]["fit_intercept"]
                        if fit =='True':
                            f1=True
                        elif fit =='False':
                            f1=False
                        copy_X=con["lr_regression"]["copy_X"]
                        if  copy_X=='True':
                            c1=True
                        elif copy_X =='False':
                            c1=False
                        positive=con["lr_regression"]["positive"]
                        if  positive =='True':
                            p1=True
                        elif positive =='False':
                            p1=False
                        n1=con["lr_regression"]["n_jobs"]
                        if n1 == 'None':
                            n2=None
                        elif n1.isdigit():
                            n2=int(n1)
                        else:
                            d1={'Status': '0',"Error":'n_jobs parameter should be None or integer','Error_code': 'A607'}
                            return d1
                        lr_regression = LinearRegression(
                            fit_intercept=f1,
                            copy_X=c1,
                            n_jobs=n2,
                            positive=p1
                        )
                        pipe = make_pipeline(StandardScaler(), lr_regression)
                        kfd = KFold(n_splits=10, random_state=1, shuffle=True)
                        for train_index, test_index in kfd.split(X, Y):
                            x_train_fold, x_test_fold = X[train_index], X[test_index]
                            y_train_fold, y_test_fold = Y[train_index], Y[test_index]
                        # model:
                        pipeline1 = pipe.fit(x_train_fold, y_train_fold)
            
                        # testing with X_train__fold:
                        y_pred = pipeline1.predict(x_train_fold)
                        r2_score1 = r2_score(y_train_fold, y_pred) 
                        mean_squared_error1 = mean_squared_error(y_train_fold, y_pred)
                        mean_absolute_error1 = mean_absolute_error(y_train_fold, y_pred)
                        Root_mean_squared_error1 = sqrt(mean_squared_error(y_train_fold, y_pred))
            
                        # testing with X_test_fold:
                        y_pred = pipeline1.predict(x_test_fold)
                        r2_score2 = r2_score(y_test_fold, y_pred) 
                        mean_squared_error2 = mean_squared_error(y_test_fold, y_pred)
                        mean_absolute_error2 = mean_absolute_error(y_test_fold, y_pred)
                        Root_mean_squared_error2 = sqrt(mean_squared_error(y_test_fold, y_pred))
                        d = {
                            "Train_info":{
                                "Train_r2_score":r2_score1,
                                "Train_mean_squared_error":mean_squared_error1,
                                "Train_mean_absolute_error":mean_absolute_error1,
                                "Train_root_mean_squared_error":Root_mean_squared_error1
                                },
        
                            "Test_info":{
                                "Test_r2_score":r2_score2,
                                "Test_mean_squared_error":mean_squared_error2,
                                "Test_mean_absolute_error":mean_absolute_error2,
                                "Test_root_mean_squared_error":Root_mean_squared_error2
                            }
                        }

                        if d["Test_info"]["Test_r2_score"] >= 0.5:
                            if con["lr_regression"]["model_generation"] == "Yes":
                                path = con["lr_regression"]["output_path"]
                                name = path + con["lr_regression"]["model_name"] + '.sav'
                                if os.path.exists(path):
                                    pickle.dump(pipeline1, open(name, 'wb'))
                                    pipeline1 = None
                                    b = "Model Generated Successfully"
                                    d1 = {"Status":'1', "Message":b, "Metrics":d, "download_path":name,"model_status":'1'}
                                    return d1
                                else:
                                    os.mkdir(path)
                                    pickle.dump(pipeline1, open(name, 'wb'))
                                    pipeline1 = None
                                    b = "Model Generated Successfully"
                                    d1 = {"Status":'1', "Message":b, "Metrics":d, "download_path":name,"model_status":'1'}
                                    return d1
                            else:
                                b = "Please Ensure Model Generation option is selected"
                                d1 = {"Status":'1', "Message":b, "Metrics":d,"model_status":'0'}
                                return d1
                        else:
                            b = "Less Efficient R2_Score"
                            d1 = {"Status":'1', "Message":b, "Metrics":d,"model_status":'0'}
                            return d1
                else:
                    d1 = {'Status': '0', 'Error': 'Null values found in data', 'Error_code': 'A234'}
                    return  d1
            else:
                d1 = {'Status': '0', 'Error': 'Unsupported Data', 'Error_code': 'A233'}
                return  d1
        else:
            d1 = {'Status': '0', 'Error': 'Insufficient Data', 'Error_code': 'A232'}
            return d1
    except Exception as e:
        logger = logging.getLogger()
        logger.critical(e)
        d1 = {'Status': '0', 'Error': str(e), 'Error code': 'A231'}
        return d1

if __name__ == "__main__":
    t, con = load()
    if t == 1:
        print(lr_reg_model(con))
    else:
        print(con)
