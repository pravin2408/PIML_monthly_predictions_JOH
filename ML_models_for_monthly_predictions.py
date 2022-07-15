#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.svm import SVR
from sklearn.gaussian_process.kernels import Matern
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer


# shifts columns of dataframe df by shift
  
def supervised(df,cols,shift):

    n_cols=len(cols)
    ad=df.iloc[:,cols]
    nms=ad.columns
    cols,names=[],[]
  
    for i in range(shift,0,-1):
        cols.append(ad.shift(i))
        names+=[('%s(t-%d)'%(nms[j],i)) for j in range(len(nms))]
    cols.append(ad.shift(0))
    names+=[('%s(t)'%(nms[j])) for j in range(len(nms))]
    agg=pd.concat(cols,axis=1)
    agg.columns=names 
    agg.dropna(inplace=True)
    return agg
    
# return a dataframe combining all dataframes with specific columns
# dataframes: List of dataframes  
# columns: List of columns
def get_dataframe(dataframes,columns):

    df=dataframes[0].loc[:,columns[0]]
    df.reset_index(inplace=True,drop=True)
    for i in range(1,len(dataframes)):
        if len(dataframes[i])>len(df):
            df1=dataframes[i].loc[1:,columns[i]]
        else:
            df1=dataframes[i].loc[:,columns[i]]
        df1.reset_index(inplace=True,drop=True) 
        df=pd.concat((df,df1),axis=1,ignore_index=True)
    df=df.dropna()
    return df

# predict output using model mdl on df and return a dataframe with columns name specified in output_columns
def get_output(df,mdl,ouptut_columns):
    df=df.values.reshape((df.shape[0],1,df.shape[1]))
    op=mdl.predict(df)
    if len(op.shape)>2:
        op=op.reshape((op.shape[0],op.shape[2]))
    else:
        op=op.reshape((op.shape[0],1))
    op=pd.DataFrame(op,columns=ouptut_columns)
    return op

#this functions removed the data  from simulated and observed data wherever the observed data contains nan
def filter_nan(s,o):
    data = np.array([s.flatten(),o.flatten()])
    data = np.transpose(data)
    return data[:,0],data[:,1]

## Evaluation metrics
def NS(s,o):
    """
    Nash Sutcliffe efficiency 
    input:
        s: simulated
        o: observed
    output:
        ns: Nash Sutcliffe efficiency
    """
    s,o = filter_nan(s,o)
    return 1 - sum((s-o)**2)/sum((o-np.mean(o))**2)
def pc_bias(s,o):
    """
    Percent Bias
    input:
        s: simulated
        o: observed
    output:
        pc_bias: percent bias
    """
    s,o = filter_nan(s,o)
    return 100.0*sum(o-s)/sum(o)
def rmse(s,o):
    """
    Root Mean Squared Error
    input:
        s: simulated
        o: observed
    output:
        rmses: root mean squared error
    """
    s,o = filter_nan(s,o)
    return np.sqrt(np.mean((s-o)**2))
def WB(s,o):
    """
    Water Balance Error
    input:
        s: simulated
        o: observed
    output:
        WB: Water Balance Error
    """

    s,o = filter_nan(s,o)
    return 1 - abs(1 - ((sum(s))/(sum(o))))

# prediction in the testing period and performance assessment using different metrics
def predict(model,model_param,test_x,test_y,name,nse,pbias,rmser,wb,model_parameters,store=True):
    predictions=model.predict(test_x)
    if store:
        d[name]=predictions
        e[nse]= NS(predictions,test_y.values)
        e[pbias]=pc_bias(predictions,test_y.values)
        e[rmser]=rmse(predictions,test_y.values)
        e[wb]=WB(predictions,test_y.values)
        e[model_parameters]=model_param
    print(NS(predictions,test_y.values))
    print(pc_bias(predictions,test_y.values))
    print(rmse(predictions,test_y.values))
    print(WB(predictions,test_y.values))
    
# Custom scorer function
def my_custom_loss_func(s,o):
    s,o = filter_nan(s,o)
    return sum((s-o)**2)/sum((o-np.mean(o))**2)

custom_scorer = make_scorer(my_custom_loss_func, greater_is_better=False)

# SVR model for Q prediction
def model_svr(train_dfx,train_dfy):
    c_value = np.linspace(1000, 50000, 50)
    tuned_svr = {'C': c_value,'kernel': ['rbf', 'poly', 'sigmoid']}
    n_folds = 5
    svr = SVR(gamma='scale', coef0=0.0, tol=0.01, epsilon=0.1, shrinking=True, cache_size=200, verbose=2, max_iter=- 1)
    model_svr = GridSearchCV(estimator = svr, param_grid = tuned_svr, scoring = custom_scorer, cv = n_folds,  n_jobs=48, verbose = 2)
    model_svr.fit(train_dfx,train_dfy)
    model_param = np.array(model_svr.best_params_)
    best_grid_svr = model_svr.best_estimator_
    predict(best_grid_svr, model_param, test_dfx, test_dfy, 'Q_svr', 'NS_svr', 'PBIAS_svr', 'RMSE_svr', 'WB_svr','Model_svr')
    test_Q_svr = model_svr.predict(test_dfx)
    return test_Q_svr 


# GPR model for Q prediction
def model_gpr_mtrn(train_dfx,train_dfy):
    alphas = [1e-10]
    nu_values = np.linspace(0.01, 1.5, 150)
    length_scale_bounds=(1e-05, 100000.0)
    tuned_gpr_mtrn = {'kernel': [Matern(length_scale,length_scale_bounds, nu) for length_scale in np.linspace(0.1, 1, 10) for nu in nu_values], 'alpha':alphas}
    n_folds = 5
    gpr_mtrn = GaussianProcessRegressor(optimizer='fmin_l_bfgs_b', n_restarts_optimizer=0, normalize_y=False, copy_X_train=True, random_state=0)
    model_gpr_mtrn = GridSearchCV(estimator = gpr_mtrn, param_grid = tuned_gpr_mtrn, scoring = custom_scorer, cv = n_folds, n_jobs = 48, verbose = 2)
    model_gpr_mtrn.fit(train_dfx.values,train_dfy.values)
    model_param = np.array(model_gpr_mtrn.best_params_)
    best_grid_gpr_mtrn = model_gpr_mtrn.best_estimator_
    predict(best_grid_gpr_mtrn, model_param, test_dfx, test_dfy, 'Q_gpr_mtrn','NS_gpr', 'PBIAS_gpr', 'RMSE_gpr', 'WB_gpr','Model_gpr')
    test_Q_gpr_mtrn = model_gpr_mtrn.predict(test_dfx)
    return test_Q_gpr_mtrn


# LASSO regression model for Q prediction
def model_lasso(train_dfx,train_dfy):
    alphas = np.linspace(0.1, 1.0, 10)
    tuned_lasso = [{'alpha': alphas}]
    n_folds = 5
    lasso = Lasso(random_state=42, max_iter=1000, selection = 'cyclic')
    model_lasso = GridSearchCV(estimator = lasso, param_grid = tuned_lasso, cv = n_folds, n_jobs = 48, verbose = 2)
    model_lasso.fit(train_dfx,train_dfy)
    model_param = np.array(model_lasso.best_params_)
    best_grid_lasso = model_lasso.best_estimator_
    predict(best_grid_lasso,model_param, test_dfx, test_dfy, 'Q_lasso', 'NS_lasso', 'PBIAS_lasso', 'RMSE_lasso', 'WB_lasso','Model_lasso')
    test_Q_lasso=model_lasso.predict(test_dfx)
    return test_Q_lasso


# Ridge regression model for Q prediction
def model_ridge(train_dfx,train_dfy):
    alphas = np.linspace(0.1, 1.0, 10)
    tuned_ridge = {'alpha': alphas, 'solver':['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga']}
    n_folds = 5
    ridge = Ridge(random_state=42, max_iter=10000)
    model_ridge = GridSearchCV(estimator = ridge, param_grid = tuned_ridge, cv = n_folds, n_jobs = 48, verbose = 2)
    model_ridge.fit(train_dfx,train_dfy)
    model_param = np.array(model_ridge.best_params_)
    best_grid_ridge = model_ridge.best_estimator_
    predict(best_grid_ridge, model_param, test_dfx, test_dfy, 'Q_ridge', 'NS_ridge', 'PBIAS_ridge', 'RMSE_ridge', 'WB_ridge','Model_ridge')
    test_Q_ridge=model_ridge.predict(test_dfx)    
    return test_Q_ridge


Perform = pd.DataFrame()
df = pd.read_csv('Anandapur_ML_input_data.csv')
d={}
e={}
train_size=28*12

# Precipitation (Pptn), Average Temperature (avg_temp), Streamflow in mm (streamflow_mm)
data = df[['Pptn','avg_temp','streamflow_mm']]

# split into train and test dataset
train_data,test_data=data.iloc[:train_size,:],data.iloc[train_size:,:]
train_dfx,train_dfy=train_data.iloc[:,:2],data.iloc[:train_size,-1]
test_dfx,test_dfy=test_data.iloc[:,:2],data.iloc[train_size:,-1]
    
# LASSO
model_lasso(train_dfx,train_dfy)

# Ridge
model_ridge(train_dfx,train_dfy)

# SVR
model_svr(train_dfx.values,train_dfy.values)

# GPR
model_gpr_mtrn(train_dfx,train_dfy)

# convert 2-d output to 1-d
for i in d.keys():
    d[i]=d[i].ravel()
Results=pd.DataFrame(d)

for j in e.keys():
    e[j]=e[j].ravel()
Results=pd.DataFrame(d)
Performance= pd.DataFrame(e)

# save model output in .csv
Results.to_csv('ML_model_test_outputs.csv')

# Performance in testing for Q with best model hyperparameters
Perform = Perform.append(Performance,ignore_index=True)
print(Perform)



