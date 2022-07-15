#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# In[3]:


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


# In[4]:


# generating training output for ET
def predict_train_ET(model,model_param,train_x,train_y,name,nse,pbias,rmser,wb,model_parameters, store=True):
    predictions=model.predict(train_x)
    if store:
        f[name]=predictions
        g[nse]= NS(predictions,train_y.values)
        g[pbias]=pc_bias(predictions,train_y.values)
        g[rmser]=rmse(predictions,train_y.values)
        g[wb]=WB(predictions,train_y.values)
        g[model_parameters]=model_param
    print(NS(predictions,train_y.values))
    print(pc_bias(predictions,train_y.values))
    print(rmse(predictions,train_y.values))
    print(WB(predictions,train_y.values))


# In[5]:


# generating training output for Q
def predict_train_Q(model,model_param,train_x,train_y,name,nse,pbias,rmser,wb,model_parameters, store=True):
    predictions=model.predict(train_x)
    if store:
        m[name]=predictions
        n[nse]= NS(predictions,train_y.values)
        n[pbias]=pc_bias(predictions,train_y.values)
        n[rmser]=rmse(predictions,train_y.values)
        n[wb]=WB(predictions,train_y.values)
        n[model_parameters]=model_param
    print(NS(predictions,train_y.values))
    print(pc_bias(predictions,train_y.values))
    print(rmse(predictions,train_y.values))
    print(WB(predictions,train_y.values))


# In[6]:


# SVR model for ET prediction
def model_svr_ET(train_dfx_ET,train_dfy_ET, test_dfx_Q):
    c_value = np.linspace(1, 5, 5)
    tuned_svr = {'C': c_value,'kernel': ['rbf', 'poly', 'sigmoid']}
    n_folds = 5
    svr = SVR(gamma='scale', coef0=0.0, tol=0.01, epsilon=0.1, shrinking=True, cache_size=200, verbose=0, max_iter=- 1)
    model_svr = GridSearchCV(estimator = svr, param_grid = tuned_svr, scoring = custom_scorer, cv = n_folds,  n_jobs=24, verbose = 2)
    model_svr.fit(train_dfx_ET,train_dfy_ET)
    model_param = np.array(model_svr.best_params_)
    best_grid_svr = model_svr.best_estimator_
    predict(best_grid_svr, model_param, test_dfx_ET, test_dfy_ET, 'ET_svr', 'NS_svr_ET', 'PBIAS_svr_ET', 'RMSE_svr_ET', 'WB_svr_ET','Model_svr_ET')
    predict_train_ET(best_grid_svr,model_param, train_dfx_ET, train_dfy_ET, 'ET_svr_train', 'NS_svr_ET', 'PBIAS_svr_ET', 'RMSE_svr_ET', 'WB_svr_ET','Model_svr_ET')
    test_ET_svr = model_svr.predict(test_dfx_ET)
    test_dfx_Q['ET_sim'] = test_ET_svr
    train_ET_svr=model_svr.predict(train_dfx_ET)
    train_dfx_Q['ET(t)'] = train_ET_svr
    return test_ET_svr, train_ET_svr


# In[7]:


# SVR model for Q prediction
def model_svr_Q(train_dfx_Q,train_dfy_Q):
    c_value = np.linspace(1, 5, 5)
    tuned_svr = {'C': c_value,'kernel': ['rbf', 'poly', 'sigmoid']}
    n_folds = 5
    svr = SVR(gamma='scale', coef0=0.0, tol=0.01, epsilon=0.1, shrinking=True, cache_size=200, verbose=0, max_iter=- 1)
    model_svr = GridSearchCV(estimator = svr, param_grid = tuned_svr, scoring = custom_scorer, cv = n_folds,  n_jobs=24, verbose = 2)
    model_svr.fit(train_dfx_Q,train_dfy_Q)
    model_param = np.array(model_svr.best_params_)
    best_grid_svr = model_svr.best_estimator_
    predict(best_grid_svr, model_param, test_dfx_Q, test_dfy_Q, 'Q_svr', 'NS_svr_Q', 'PBIAS_svr_Q', 'RMSE_svr_Q', 'WB_svr_Q','Model_svr_Q')
    predict_train_Q(best_grid_svr,model_param, train_dfx_Q, train_dfy_Q, 'Q_svr_train', 'NS_svr_Q', 'PBIAS_svr_Q', 'RMSE_svr_Q', 'WB_svr_Q','Model_svr_Q')
    test_Q_svr = model_svr.predict(test_dfx_Q)
    return test_Q_svr 


# In[8]:


# GPR model for ET prediction
def model_gpr_mtrn_ET(train_dfx_ET,train_dfy_ET, test_dfx_Q):
    alphas = [1e-10]
    nu_values = np.linspace(0.01, 0.1, 10)
    length_scale_bounds=(1e-05, 100000.0)
    tuned_gpr_mtrn = {'kernel': [Matern(length_scale,length_scale_bounds, nu) for length_scale in np.linspace(0.1, 1, 10) for nu in nu_values], 'alpha':alphas}
    n_folds = 5
    gpr_mtrn = GaussianProcessRegressor(optimizer='fmin_l_bfgs_b', n_restarts_optimizer=0, normalize_y=False, copy_X_train=True, random_state=0)
    model_gpr_mtrn = GridSearchCV(estimator = gpr_mtrn, param_grid = tuned_gpr_mtrn, scoring = custom_scorer, cv = n_folds, n_jobs = 24, verbose = 0)
    model_gpr_mtrn.fit(train_dfx_ET,train_dfy_ET)
    model_param = np.array(model_gpr_mtrn.best_params_)
    best_grid_gpr_mtrn = model_gpr_mtrn.best_estimator_
    predict(best_grid_gpr_mtrn, model_param, test_dfx_ET, test_dfy_ET, 'ET_gpr_mtrn','NS_gpr_ET', 'PBIAS_gpr_ET', 'RMSE_gpr_ET', 'WB_gpr_ET','Model_gpr_ET')
    predict_train_ET(best_grid_gpr_mtrn,model_param, train_dfx_ET, train_dfy_ET, 'ET_gpr_mtrn_train', 'NS_gpr_mtrn_ET', 'PBIAS_gpr_mtrn_ET', 'RMSE_gpr_mtrn_ET', 'WB_gpr_mtrn_ET','Model_gpr_mtrn_ET')
    test_ET_gpr_mtrn = model_gpr_mtrn.predict(test_dfx_ET)
    train_ET_gpr_mtrn=model_gpr_mtrn.predict(train_dfx_ET)
    train_dfx_Q['ET(t)'] = train_ET_gpr_mtrn
    return test_ET_gpr_mtrn, train_ET_gpr_mtrn


# In[9]:


# GPR model for Q prediction
def model_gpr_mtrn_Q(train_dfx_Q,train_dfy_Q):
    alphas = [1e-10]
    nu_values = np.linspace(0.01, 0.1, 10)
    length_scale_bounds=(1e-05, 100000.0)
    tuned_gpr_mtrn = {'kernel': [Matern(length_scale,length_scale_bounds, nu) for length_scale in np.linspace(0.1, 1, 10) for nu in nu_values], 'alpha':alphas}
    n_folds = 5
    gpr_mtrn = GaussianProcessRegressor(optimizer='fmin_l_bfgs_b', n_restarts_optimizer=0, normalize_y=False, copy_X_train=True, random_state=0)
    model_gpr_mtrn = GridSearchCV(estimator = gpr_mtrn, param_grid = tuned_gpr_mtrn, scoring = custom_scorer, cv = n_folds, n_jobs = 24, verbose = 0)
    model_gpr_mtrn.fit(train_dfx_Q,train_dfy_Q)
    model_param = np.array(model_gpr_mtrn.best_params_)
    best_grid_gpr_mtrn = model_gpr_mtrn.best_estimator_
    predict(best_grid_gpr_mtrn, model_param, test_dfx_Q, test_dfy_Q, 'Q_gpr_mtrn','NS_gpr_Q', 'PBIAS_gpr_Q', 'RMSE_gpr_Q', 'WB_gpr_Q','Model_gpr_Q')
    predict_train_Q(best_grid_gpr_mtrn,model_param, train_dfx_Q, train_dfy_Q, 'Q_gpr_mtrn_train', 'NS_gpr_mtrn_Q', 'PBIAS_gpr_mtrn_Q', 'RMSE_gpr_mtrn_Q', 'WB_gpr_mtrn_Q','Model_gpr_mtrn_Q')
    test_Q_gpr_mtrn = model_gpr_mtrn.predict(test_dfx_Q)
    return test_Q_gpr_mtrn


# In[10]:


# LASSO regression model for ET prediction
def model_lasso_ET(train_dfx_ET, train_dfy_ET, test_dfx_Q):
    alphas = np.linspace(1.0, 1.0, 1)
    tuned_lasso = [{'alpha': alphas}]
    n_folds = 5
    lasso = Lasso(random_state=42, max_iter=1000, selection = 'cyclic')
    model_lasso = GridSearchCV(estimator = lasso, param_grid = tuned_lasso, cv = n_folds, n_jobs = 24, verbose = 0)
    model_lasso.fit(train_dfx_ET,train_dfy_ET)
    model_param = np.array(model_lasso.best_params_)
    best_grid_lasso = model_lasso.best_estimator_
    predict(best_grid_lasso,model_param, test_dfx_ET, test_dfy_ET, 'ET_lasso', 'NS_lasso_ET', 'PBIAS_lasso_ET', 'RMSE_lasso_ET', 'WB_lasso_ET','Model_lasso_ET')
    predict_train_ET(best_grid_lasso,model_param, train_dfx_ET, train_dfy_ET, 'ET_lasso_train', 'NS_lasso_ET', 'PBIAS_lasso_ET', 'RMSE_lasso_ET', 'WB_lasso_ET','Model_lasso_ET')
    test_ET_lasso=model_lasso.predict(test_dfx_ET)
    test_dfx_Q['ET_sim'] = test_ET_lasso
    train_ET_lasso=model_lasso.predict(train_dfx_ET)
    train_dfx_Q['ET(t)'] = train_ET_lasso
    return test_ET_lasso, train_ET_lasso


# In[11]:


# LASSO regression model for Q prediction
def model_lasso_Q(train_dfx_Q,train_dfy_Q):
    alphas = np.linspace(0.1, 1.0, 10)
    tuned_lasso = [{'alpha': alphas}]
    n_folds = 5
    lasso = Lasso(random_state=42, max_iter=1000, selection = 'cyclic')
    model_lasso = GridSearchCV(estimator = lasso, param_grid = tuned_lasso, cv = n_folds, n_jobs = 24, verbose = 0)
    model_lasso.fit(train_dfx_Q,train_dfy_Q)
    model_param = np.array(model_lasso.best_params_)
    best_grid_lasso = model_lasso.best_estimator_
    predict(best_grid_lasso,model_param, test_dfx_Q, test_dfy_Q, 'Q_lasso', 'NS_lasso_Q', 'PBIAS_lasso_Q', 'RMSE_lasso_Q', 'WB_lasso_Q','Model_lasso_Q')
    predict_train_Q(best_grid_lasso,model_param, train_dfx_Q, train_dfy_Q, 'Q_lasso_train', 'NS_lasso_Q', 'PBIAS_lasso_Q', 'RMSE_lasso_Q', 'WB_lasso_Q','Model_lasso_Q')
    test_Q_lasso=model_lasso.predict(test_dfx_Q)
    return test_Q_lasso


# In[12]:


# Ridge regression model for ET prediction
def model_ridge_ET(train_dfx_ET, train_dfy_ET, test_dfx_Q):
    alphas = np.linspace(1.0, 1.0, 1)
    tuned_ridge = {'alpha': alphas, 'solver':['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga']}
    n_folds = 5
    ridge = Ridge(random_state=42, max_iter=10000)
    model_ridge = GridSearchCV(estimator = ridge, param_grid = tuned_ridge, cv = n_folds, n_jobs = 24, verbose = 0)
    model_ridge.fit(train_dfx_ET,train_dfy_ET)
    model_param = np.array(model_ridge.best_params_)
    best_grid_ridge = model_ridge.best_estimator_
    predict(best_grid_ridge, model_param, test_dfx_ET, test_dfy_ET, 'ET_ridge', 'NS_ridge_ET', 'PBIAS_ridge_ET', 'RMSE_ridge_ET', 'WB_ridge_ET','Model_ridge_ET')
    predict_train_ET(best_grid_ridge,model_param, train_dfx_ET, train_dfy_ET, 'ET_ridge_train', 'NS_ridge_ET', 'PBIAS_ridge_ET', 'RMSE_ridge_ET', 'WB_ridge_ET','Model_ridge_ET')
    test_ET_ridge=model_ridge.predict(test_dfx_ET)
    test_dfx_Q['ET_sim'] = test_ET_ridge
    train_ET_ridge=model_ridge.predict(train_dfx_ET)
    train_dfx_Q['ET(t)'] = train_ET_ridge
    return test_ET_ridge, train_ET_ridge


# In[13]:


# Ridge regression model for Q prediction
def model_ridge_Q(train_dfx_Q,train_dfy_Q):
    alphas = np.linspace(0.1, 1.0, 10)
    tuned_ridge = {'alpha': alphas, 'solver':['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga']}
    n_folds = 5
    ridge = Ridge(random_state=42, max_iter=10000)
    model_ridge = GridSearchCV(estimator = ridge, param_grid = tuned_ridge, cv = n_folds, n_jobs = 24, verbose = 0)
    model_ridge.fit(train_dfx_Q,train_dfy_Q)
    model_param = np.array(model_ridge.best_params_)
    best_grid_ridge = model_ridge.best_estimator_
    predict(best_grid_ridge, model_param, test_dfx_Q, test_dfy_Q, 'Q_ridge', 'NS_ridge_Q', 'PBIAS_ridge_Q', 'RMSE_ridge_Q', 'WB_ridge_Q','Model_ridge_Q')
    predict_train_Q(best_grid_ridge,model_param, train_dfx_Q, train_dfy_Q, 'Q_ridge_train', 'NS_ridge_Q', 'PBIAS_ridge_Q', 'RMSE_ridge_Q', 'WB_ridge_Q','Model_ridge_Q')
    test_Q_ridge=model_ridge.predict(test_dfx_Q)    
    return test_Q_ridge


# In[14]:


Perform = pd.DataFrame()
Perform_train_ET = pd.DataFrame()
Perform_train_Q = pd.DataFrame()
df = pd.read_csv('Anandapur_PIML_input_data.csv')
d={}
e={}
f={}
g={}
m={}
n={}
train_size=(28*12)

# Precipitation (Pptn), Potential Evapotranspiration (PET), Soil Moisture (S), Groundwater Storage (G), Actual Evapotranspiration (ET), Streamflow in mm (streamflow_mm)
data = df[['Pptn','PET','S','G','ET','streamflow_mm']]

# split into train and test dataset
train_data,test_data=data.iloc[:train_size,:],data.iloc[train_size:,:]
supervised_train_data=supervised(train_data,[0,1,2,3,4,5],1)
supervised_test_data=supervised(test_data,[0,1,2,3,4,5],1)
train_dfx_ET,train_dfy_ET=get_dataframe([supervised_train_data],[['Pptn(t)','PET(t)','S(t-1)']]),get_dataframe([supervised_train_data],[['ET(t)']])
test_dfx_ET,test_dfy_ET=get_dataframe([supervised_test_data],[['Pptn(t)','PET(t)','S(t-1)']]),get_dataframe([supervised_test_data],[['ET(t)']])
train_dfx_Q,train_dfy_Q=get_dataframe([supervised_train_data],[['Pptn(t)','S(t)','S(t-1)','G(t)','G(t-1)','ET(t)']]),get_dataframe([supervised_train_data],[['streamflow_mm(t)']])
test_dfx_Q,test_dfy_Q=get_dataframe([supervised_test_data],[['Pptn(t)','S(t)','S(t-1)','G(t)','G(t-1)']]),get_dataframe([supervised_test_data],[['streamflow_mm(t)']])
train_dfx_ET_gpr,train_dfy_ET_gpr=get_dataframe([supervised_train_data],[['Pptn(t)','PET(t)','S(t-1)']]),get_dataframe([supervised_train_data],[['ET(t)']])

# LASSO
model_lasso_ET(train_dfx_ET,train_dfy_ET,test_dfx_Q)
model_lasso_Q(train_dfx_Q,train_dfy_Q)

# Ridge
model_ridge_ET(train_dfx_ET,train_dfy_ET,test_dfx_Q)
model_ridge_Q(train_dfx_Q,train_dfy_Q)

# SVR
model_svr_ET(train_dfx_ET,train_dfy_ET, test_dfx_Q)
model_svr_Q(train_dfx_Q,train_dfy_Q)

# GPR
model_gpr_mtrn_ET(train_dfx_ET_gpr,train_dfy_ET_gpr,test_dfx_Q)
model_gpr_mtrn_Q(train_dfx_Q,train_dfy_Q)

#    convert 2-d output to 1-d
for i in d.keys():
    d[i]=d[i].ravel()
Results=pd.DataFrame(d)
for j in e.keys():
    e[j]=e[j].ravel()
Performance= pd.DataFrame(e)
for k in f.keys():
    f[k]=f[k].ravel()
Results_train_ET=pd.DataFrame(f)
for l in g.keys():
    g[l]=g[l].ravel()
Performance_train_ET= pd.DataFrame(g)
for p in m.keys():
    m[p]=m[p].ravel()
Results_train_Q=pd.DataFrame(m)
for q in n.keys():
    n[q]=n[q].ravel()
Performance_train_Q= pd.DataFrame(n)

# save model output in .csv
Results.to_csv('PIML_model_test_outputs.csv')

# Performance in testing for ET and Q with best model hyperparameters
Perform = Perform.append(Performance,ignore_index=True)
print(Perform)

# Performance in training for ET
Perform_train_ET = Perform_train_ET.append(Performance_train_ET,ignore_index=True)
print(Perform_train_ET)

# Performance in training for Q
Perform_train_Q = Perform_train_Q.append(Performance_train_Q,ignore_index=True)
print(Perform_train_Q)

