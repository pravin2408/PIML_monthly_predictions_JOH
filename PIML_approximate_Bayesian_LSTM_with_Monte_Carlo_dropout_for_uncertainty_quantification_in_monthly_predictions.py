#!/usr/bin/env python
# coding: utf-8

import numpy as np
import random
import tensorflow as tf
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential, Input, Model
from keras import optimizers
from keras.layers import LSTM, Dense, Dropout
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error,r2_score
from math import sqrt


# fix the figure size and axes grid
mpl.rcParams['figure.figsize'] =  (6,6)
mpl.rcParams['axes.grid'] = False


##shifts columns of dataframe df by shift
def supervised(df,cols,shift):
    n_cols=len(cols)
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

#this functions removed the data  from simulated and observed data wherever the observed data contains nan
def filter_nan(s,o):
    data = np.array([s.flatten(),o.flatten()])
    data = np.transpose(data)
    return data[:,0],data[:,1]

## Evaluation metrics
def NS(s,o):
    """"
    #Nash Sutcliffe efficiency 
    #input:
        #s: simulated
        #o: observed
    #output:
        #NS: Nash Sutcliffe efficient 
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
        rmse: root mean squared error
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


perform = pd.DataFrame()
perform_train = pd.DataFrame()
perform_train["NSE_cal_ET"] = ""
perform_train["PBIAS_cal_ET"] = ""
perform_train["RMSE_cal_ET"] = ""
perform_train["WB_cal_ET"] = ""
perform_train["NSE_cal_Q"] = ""
perform_train["PBIAS_cal_Q"] = ""
perform_train["RMSE_cal_Q"] = ""
perform_train["WB_cal_Q"] = ""
perform["NSE_val_ET"] = ""
perform["PBIAS_val_ET"] = ""
perform["RMSE_val_ET"] = ""
perform["WB_val_ET"] = ""
perform['Batch_size_ET'] = ""
perform["NSE_val_Q"] = ""
perform["PBIAS_val_Q"] = ""
perform["RMSE_val_Q"] = ""
perform["WB_val_Q"] = ""
perform['Batch_size_Q'] = ""
# Hyperparameters
dropout = 0.1
Nepoch = 80
hidden_units = 50
batch_size_value = 6

df = pd.read_csv('Anandapur_PIML_input_data.csv')
data = df[['Pptn','PET','ET','S','G',"streamflow_mm"]]
data_train = df[['ET','streamflow_mm']]
nmonths = 2  
nsteps = 1 
ninputs_ET = 3
nobs_ET = nmonths * ninputs_ET
Ntest = 28*12-1
ninputs_Q = 4
nobs_Q = nmonths * ninputs_Q

# Number of Monte Carlo simulations 
Nmc = 100

output_df = pd.DataFrame((data.iloc[Ntest+2:,2]).values)
train_df_ET = pd.DataFrame((data_train.iloc[1:Ntest+1,0]).values)
train_df_Q = pd.DataFrame((data_train.iloc[1:Ntest+1,1]).values)
output_df_Q = pd.DataFrame((data.iloc[Ntest+2:,5]).values)

## ET prediction        
reframed = supervised(data,[0,1,2,3,4,5],1)
print('Shape of supervised dataset: ', np.shape(reframed))
reframed_new_ET = reframed[['Pptn(t-1)', 'PET(t)','PET(t-1)','S(t-1)','S(t)','Pptn(t)', 'ET(t)']]
reframed_new_ET[['Pptn(t-1)', 'PET(t-1)','S(t)']] = 0
XYdata_ET = reframed_new_ET.values

# split into train and test dataset
XYtrain_ET = XYdata_ET[:Ntest, :]
XYtest_ET = XYdata_ET[Ntest+1:, :]
yobs_train_ET = XYdata_ET[:Ntest, -nsteps]
yobs_test_ET = XYdata_ET[Ntest+1:, -nsteps]
print('shape of yobs_train_ET and yobs_test_ET is ', yobs_train_ET.shape, yobs_test_ET.shape)
print('min and max of yobs_test_ET', np.min(yobs_test_ET), np.max(yobs_test_ET))

# scale training and testing data
scaler = MinMaxScaler(feature_range=(0, 1))
scaledXYtrain_ET = scaler.fit_transform(XYtrain_ET)
scaledXYtest_ET = scaler.transform(XYtest_ET)
print('shape of scaledXYtrain and scaledXYtest is ', scaledXYtrain_ET.shape, scaledXYtest_ET.shape)

# split into input and outputs
train_X_ET, train_y_ET = scaledXYtrain_ET[:, :nobs_ET], scaledXYtrain_ET[:, -nsteps]
test_X_ET = scaledXYtest_ET[:, :nobs_ET]

print('shape of train_X_ET, train_y_ET, and test_X_ET: ', train_X_ET.shape, train_y_ET.shape, test_X_ET.shape)

# reshape input to be 3D [samples, timesteps, features]
train_X_ET = train_X_ET.reshape((train_X_ET.shape[0], nmonths, ninputs_ET))
test_X_ET = test_X_ET.reshape((test_X_ET.shape[0], nmonths, ninputs_ET))
print('shape of train_X_ET and test_X_ET in 3D: ', train_X_ET.shape, test_X_ET.shape)

rmse_train = np.zeros((Nepoch,1))
r2_train = np.zeros((Nepoch,1))
nse_train = np.zeros((Nepoch,1))
ave_std_train = np.zeros((Nepoch,1))
rmse_test = np.zeros((Nepoch,1))
r2_test = np.zeros((Nepoch,1))
nse_test = np.zeros((Nepoch,1))
ave_std_test = np.zeros((Nepoch,1))

input_shape=(train_X_ET.shape[1], train_X_ET.shape[2])

# define LSTM model with recurrent dropout
inp = Input(input_shape)
tf.random.set_seed(1234)
x = LSTM(hidden_units,recurrent_dropout=dropout)(inp,training=True)
out = Dense(nsteps, activation = 'relu')(x)
model = Model(inputs=inp, outputs=out)
model.compile(optimizer='adam',loss='mse') 

# fit LSTM model
for i in range(Nepoch):       
    model.fit(train_X_ET, train_y_ET, epochs=1, batch_size=batch_size_value, validation_split=0.2, verbose=2)

    print('---------------- Epoch %d -------------' %i)
 
    mc_train = []
    mc_test = []
    # run Monte Carlo loop
    for imc in range(Nmc):
        # evaluate training data
        yhat = model.predict(train_X_ET)
        train_X_ET = train_X_ET.reshape((train_X_ET.shape[0], nobs_ET))
        inv_yhat = np.concatenate((train_X_ET,yhat), axis=1)
        inv_yhat = scaler.inverse_transform(inv_yhat)
        NNytrain = inv_yhat[:,-nsteps]
        mc_train.append(NNytrain)
        train_X_ET = train_X_ET.reshape((train_X_ET.shape[0], nmonths, ninputs_ET))

        # evaluate testing data
        yhat_test = model.predict(test_X_ET)
        test_X_ET = test_X_ET.reshape((test_X_ET.shape[0], nobs_ET))
        tmp_test = np.concatenate((test_X_ET, yhat_test), axis=1)
        inv_yhat_test = scaler.inverse_transform(tmp_test)
        NNytest = inv_yhat_test[:,-nsteps]
        mc_test.append(NNytest)
        test_X_ET = test_X_ET.reshape((test_X_ET.shape[0], nmonths, ninputs_ET))
    mc_train_ET = np.array(mc_train)
    print('shape of mc_train is', mc_train_ET.shape)
    std_train = mc_train_ET.std(axis=0)
    ave_std_train[i] = std_train.mean()
    mean_train_ET = mc_train_ET.mean(axis=0)
    rmse_train[i] = sqrt(mean_squared_error(yobs_train_ET, mean_train_ET))
    r2_train[i] = r2_score(yobs_train_ET, mean_train_ET)
    (s_train,o_train) = (mean_train_ET, yobs_train_ET)
    nse_train[i] = 1 - sum((s_train-o_train)**2)/sum((o_train-np.mean(o_train))**2)
    print('Training -------: rmse, r2, and nse are %6.3f, %6.3f, and %6.3f' % (rmse_train[i],r2_train[i],nse_train[i]))
    (s_ET_cal,o_ET_cal) = (mean_train_ET, yobs_train_ET)
    NS_ET_cal = 1 - sum((s_ET_cal-o_ET_cal)**2)/sum((o_ET_cal-np.mean(o_ET_cal))**2)
    PBIAS_ET_cal = pc_bias(s_ET_cal,o_ET_cal)
    RMSE_ET_cal = rmse(s_ET_cal,o_ET_cal)
    WB_ET_cal = 1 - abs(1 - ((sum(s_ET_cal))/(sum(o_ET_cal))))
    mc_test_ET = np.array(mc_test)
    print('shape of mc_test is', mc_test_ET.shape)
    std_test = mc_test_ET.std(axis=0)
    ave_std_test[i] = std_test.mean()
    mean_test_ET = mc_test_ET.mean(axis=0)
    (s_test,o_test) = (mean_test_ET, yobs_test_ET)
    rmse_test[i] = sqrt(mean_squared_error(yobs_test_ET, mean_test_ET))
    r2_test[i] = r2_score(yobs_test_ET, mean_test_ET)
    nse_test[i] = 1 - sum((s_test-o_test)**2)/sum((o_test-np.mean(o_test))**2)
    print('Testing--------------------: rmse, r2, and nse are %6.3f, %6.3f, and %6.3f' % (rmse_test[i],r2_test[i],nse_test[i]))

# performance evaluation for mean ET     
(s_ET,o_ET) = (mean_test_ET, yobs_test_ET)
NS_ET = 1 - sum((s_ET-o_ET)**2)/sum((o_ET-np.mean(o_ET))**2)
PBIAS_ET = pc_bias(s_ET,o_ET)
RMSE_ET = rmse(s_ET,o_ET)
WB_ET = 1 - abs(1 - ((sum(s_ET))/(sum(o_ET))))

## Q prediction  
reframed = supervised(data,[0,1,2,3,4,5],1)
print('Shape of supervised datasQ: ', np.shape(reframed))
reframed_new_Q = reframed[[ 'ET(t)','ET(t-1)','S(t-1)','S(t)','Pptn(t)','Pptn(t-1)','G(t-1)','G(t)', 'streamflow_mm(t)']]
reframed_new_Q[['Pptn(t-1)', 'ET(t-1)']] = 0
XYdata_Q = reframed_new_Q.values

# split into train and test dataset
XYtrain_Q = XYdata_Q[:Ntest, :]
XYtest_Q = XYdata_Q[Ntest+1:, :]
XYtrain_Q[:,0] = mean_train_ET
XYtest_Q[:,0] = mean_test_ET
yobs_train_Q = XYdata_Q[:Ntest, -nsteps]
yobs_test_Q = XYdata_Q[Ntest+1:, -nsteps]
print('shape of yobs_train_Q and yobs_test_Q is ', yobs_train_Q.shape, yobs_test_Q.shape)
print('min and max of yobs_test_Q', np.min(yobs_test_Q), np.max(yobs_test_Q))

# scale training and testing data
scaler = MinMaxScaler(feature_range=(0, 1))
scaledXYtrain_Q = scaler.fit_transform(XYtrain_Q)
scaledXYtest_Q = scaler.transform(XYtest_Q)
print('shape of scaledXYtrain and scaledXYtest is ', scaledXYtrain_Q.shape, scaledXYtest_Q.shape)

# split into input and outputs
train_X_Q, train_y_Q = scaledXYtrain_Q[:, :nobs_Q], scaledXYtrain_Q[:, -nsteps]
test_X_Q = scaledXYtest_Q[:, :nobs_Q]
print('shape of train_X_Q, train_y_Q, and test_X_Q: ', train_X_Q.shape, train_y_Q.shape, test_X_Q.shape)

# reshape input to be 3D [samples, timesteps, features]
train_X_Q = train_X_Q.reshape((train_X_Q.shape[0], nmonths, ninputs_Q))
test_X_Q = test_X_Q.reshape((test_X_Q.shape[0], nmonths, ninputs_Q))
print('shape of train_X_Q and test_X_Q in 3D: ', train_X_Q.shape, test_X_Q.shape)
input_shape_Q=(train_X_Q.shape[1], train_X_Q.shape[2])

# define LSTM model with recurrent dropout
inp_Q = Input(input_shape_Q)
tf.random.set_seed(1234)
x_Q = LSTM(hidden_units,recurrent_dropout=dropout)(inp_Q,training=True)
out_Q = Dense(nsteps, activation = 'relu')(x_Q)
model_Q = Model(inputs=inp_Q, outputs=out_Q)
model_Q.compile(optimizer='adam',loss='mse') 

# fit LSTM model
for i in range(Nepoch):       
    model_Q.fit(train_X_Q, train_y_Q, epochs=1, batch_size=batch_size_value, validation_split=0.2, verbose=42)
    mc_train_Q = []
    mc_test_Q = []
    # run Monte Carlo loop
    for imc in range(Nmc):
        # evaluate training data
        yhat = model_Q.predict(train_X_Q)
        train_X_Q = train_X_Q.reshape((train_X_Q.shape[0], nobs_Q))
        inv_yhat = np.concatenate((train_X_Q,yhat), axis=1)
        inv_yhat = scaler.inverse_transform(inv_yhat)
        NNytrain = inv_yhat[:,-nsteps]
        mc_train_Q.append(NNytrain)
        train_X_Q = train_X_Q.reshape((train_X_Q.shape[0], nmonths, ninputs_Q))

        # evaluate testing data
        yhat_test = model_Q.predict(test_X_Q)
        test_X_Q = test_X_Q.reshape((test_X_Q.shape[0], nobs_Q))
        tmp_test = np.concatenate((test_X_Q, yhat_test), axis=1)
        inv_yhat_test = scaler.inverse_transform(tmp_test)
        NNytest = inv_yhat_test[:,-nsteps]
        mc_test_Q.append(NNytest)
        test_X_Q = test_X_Q.reshape((test_X_Q.shape[0], nmonths, ninputs_Q))
    mc_train_Q = np.array(mc_train_Q)
    print('shape of mc_train is', mc_train_Q.shape)
    std_train = mc_train_Q.std(axis=0)
    ave_std_train[i] = std_train.mean()
    mean_train_Q = mc_train_Q.mean(axis=0)
    rmse_train[i] = sqrt(mean_squared_error(yobs_train_Q, mean_train_Q))
    r2_train[i] = r2_score(yobs_train_Q, mean_train_Q)
    (s_train,o_train) = (mean_train_Q, yobs_train_Q)
    nse_train[i] = 1 - sum((s_train-o_train)**2)/sum((o_train-np.mean(o_train))**2)
    print('Training -------: rmse, r2, and nse are %6.3f, %6.3f, and %6.3f' % (rmse_train[i],r2_train[i],nse_train[i]))
    (s_Q_cal,o_Q_cal) = (mean_train_Q, yobs_train_Q)
    NS_Q_cal = 1 - sum((s_Q_cal-o_Q_cal)**2)/sum((o_Q_cal-np.mean(o_Q_cal))**2)
    PBIAS_Q_cal = pc_bias(s_Q_cal,o_Q_cal)
    RMSE_Q_cal = rmse(s_Q_cal,o_Q_cal)
    WB_Q_cal = 1 - abs(1 - ((sum(s_Q_cal))/(sum(o_Q_cal))))
    mc_test2 = np.array(mc_test_Q)
    print('shape of mc_test is', mc_test2.shape)
    std_test = mc_test2.std(axis=0)
    ave_std_test[i] = std_test.mean()
    mean_test = mc_test2.mean(axis=0)
    (s_test,o_test) = (mean_test, yobs_test_Q)
    rmse_test[i] = sqrt(mean_squared_error(yobs_test_Q, mean_test))
    r2_test[i] = r2_score(yobs_test_Q, mean_test)
    nse_test[i] = 1 - sum((s_test-o_test)**2)/sum((o_test-np.mean(o_test))**2)
    print('Testing--------------------: rmse, r2, and nse are %6.3f, %6.3f, and %6.3f' % (rmse_test[i],r2_test[i],nse_test[i]))
               
# performance evaluation for mean Q 
(s,o) = (mean_test, yobs_test_Q)
NS_Q = 1 - sum((s-o)**2)/sum((o-np.mean(o))**2)
PBIAS_Q = pc_bias(s,o)
RMSE_Q = rmse(s,o)
WB_Q = 1 - abs(1 - ((sum(s))/(sum(o))))

# testing performance for ET and Q
perform = perform.append({'Dropout':dropout,'Hidden_units':hidden_units,'Epochs':Nepoch ,'Batch_size_ET':batch_size_value,'NSE_val_ET':NS_ET,'PBIAS_val_ET':PBIAS_ET ,'RMSE_val_ET':RMSE_ET,'WB_val_ET':WB_ET, 'Batch_size_Q':batch_size_value,'NSE_val_Q':NS_Q,'PBIAS_val_Q':PBIAS_Q ,'RMSE_val_Q':RMSE_Q,'WB_val_Q':WB_Q},ignore_index = True)
print(perform)

# training performance for ET and Q
perform_train = perform_train.append({'Batch_size_ET':batch_size_value,'NSE_cal_ET':NS_ET_cal,'PBIAS_cal_ET':PBIAS_ET_cal ,'RMSE_cal_ET':RMSE_ET_cal,'WB_cal_ET':WB_ET_cal, 'Batch_size_Q':batch_size_value,'NSE_cal_Q':NS_Q_cal,'PBIAS_cal_Q':PBIAS_Q_cal ,'RMSE_cal_Q':RMSE_Q_cal,'WB_cal_Q':WB_Q_cal },ignore_index = True)        
print(perform_train)

# save model output in .csv (For ET)
mc_test1 = mc_test_ET.transpose()
mc_test_df1 = pd.DataFrame(mc_test1)
output_df_ET = pd.concat([output_df,mc_test_df1], axis=1)
output_df_ET.to_csv(''PIML_BLSTM_test_outputs_ET.csv'')

mc_train1 = mc_train_ET.transpose()
mc_train_df1 = pd.DataFrame(mc_train1)
train_df_ET = pd.concat([train_df_ET,mc_train_df1], axis=1)

# save model output in .csv (For Q)
mc_test2 = mc_test2.transpose()
mc_test_df2 = pd.DataFrame(mc_test2)
output_df_Q = pd.concat([output_df_Q,mc_test_df2], axis=1)
output_df_Q.to_csv(''PIML_BLSTM_test_outputs_Q.csv'')

mc_train2 = mc_train_Q.transpose()
mc_train_df2 = pd.DataFrame(mc_train2)
train_df_Q = pd.concat([train_df_Q,mc_train_df2], axis=1)



