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


# fix the figure size and axes grid
mpl.rcParams['figure.figsize'] =  (12,12)
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
perform['Dropout'] = ""
perform['Hidden_units'] = ""
perform['Epochs'] = ""
perform['Batch_size_Q'] = ""
perform['Batch_size_Q'] = ""
perform["NSE_cal_Q"] = ""
perform["PBIAS_cal_Q"] = ""
perform["RMSE_cal_Q"] = ""
perform["WB_cal_Q"] = ""
perform["NSE_val_Q"] = ""
perform["PBIAS_val_Q"] = ""
perform["RMSE_val_Q"] = ""
perform["WB_val_Q"] = ""
df = pd.read_csv('Anandapur_ML_input_data.csv')
data = df[['Pptn','avg_temp','streamflow_mm']]
nmonths = 1  
nsteps = 1 
Ntest = 28*12
ninputs_Q = 2
nobs_Q = nmonths * ninputs_Q

# dropouts
for h in range (0,4):
    dropout_rate = [0.1,0.2,0.3,0.4]
    dropout_rate_value = dropout_rate[h]
    print('dropout_rate:',dropout_rate_value)
    # epochs
    for i in range(0,10):
        n_epochs = [10,20,30,40,50,60,70,80,90,100]
        epochs_value = n_epochs[i]
        print('epochs:',epochs_value)
        # hidden units
        for j in range (0,5):
            hidden_units = [10,20,30,40,50]
            hidden_unit_value = hidden_units[j]
            print('hidden_units:',hidden_unit_value)
            # batch size
            for k in range(0,9):
                batch_size = [6,12,18,24,36,48,60,84,96]
                batch_size_value = batch_size[k]
                output_df = pd.DataFrame((data.iloc[Ntest:,2]).values)
                print('batch_size:',batch_size_value)
                reframed = supervised(data,[0,1,2],0)
                reframed_new_Q = reframed[[ 'Pptn(t)','avg_temp(t)','streamflow_mm(t)']]
                XYdata_Q = reframed_new_Q.values

                # split into train and test dataset
                XYtrain_Q = XYdata_Q[:Ntest, :]
                XYtest_Q = XYdata_Q[Ntest:, :]
                yobs_train_Q = XYdata_Q[:Ntest, -nsteps]
                yobs_test_Q = XYdata_Q[Ntest:, -nsteps]
                
                # scale training and testing data
                scaler = MinMaxScaler(feature_range=(0, 1))
                scaledXYtrain_Q = scaler.fit_transform(XYtrain_Q)
                scaledXYtest_Q = scaler.transform(XYtest_Q)

                # split into input and outputs
                train_X_Q, train_y_Q = scaledXYtrain_Q[:, :nobs_Q], scaledXYtrain_Q[:, -nsteps]
                test_X_Q = scaledXYtest_Q[:, :nobs_Q]
                    
                # reshape input to be 3D [samples, timesteps, features]
                train_X_Q = train_X_Q.reshape((train_X_Q.shape[0], nmonths, ninputs_Q))
                test_X_Q = test_X_Q.reshape((test_X_Q.shape[0], nmonths, ninputs_Q))
                
                input_shape=(train_X_Q.shape[1], train_X_Q.shape[2])
                tf.random.set_seed(1234)
                
                # define and fit LSTM model
                model = Sequential()
                model.add(LSTM(hidden_unit_value, input_shape=(train_X_Q.shape[1], train_X_Q.shape[2])))
                model.add(Dropout(dropout_rate_value))
                model.add(Dense(nsteps, activation = 'relu')) 
                model.compile(loss='mae', optimizer='adam')
                history  = model.fit(train_X_Q, train_y_Q, epochs=epochs_value, batch_size=batch_size_value, validation_split=0.2, verbose=0)
                plt.plot(history.history['loss'])
                plt.plot(history.history['val_loss'])
                plt.title('model loss')
                plt.ylabel('loss')
                plt.xlabel('epoch')
                plt.legend(['train', 'val'], loc='upper left')
                plt.show()
                
                # evaluate training data
                yhat_Q = model.predict(train_X_Q)
                train_X_Q = train_X_Q.reshape((train_X_Q.shape[0], nobs_Q))
                inv_yhat_Q = np.concatenate((train_X_Q,yhat_Q), axis=1)
                inv_yhat_Q = scaler.inverse_transform(inv_yhat_Q)
                NNytrain_Q = inv_yhat_Q[:,-nsteps]
                train_output_Q = (NNytrain_Q)
                train_X_Q = train_X_Q.reshape((train_X_Q.shape[0], nmonths, ninputs_Q))
                (s_Q_cal,o_Q_cal) = (train_output_Q, yobs_train_Q)
                NS_Q_cal = 1 - sum((s_Q_cal-o_Q_cal)**2)/sum((o_Q_cal-np.mean(o_Q_cal))**2)
                PBIAS_Q_cal = pc_bias(s_Q_cal,o_Q_cal)
                RMSE_Q_cal = rmse(s_Q_cal,o_Q_cal)
                WB_Q_cal = 1 - abs(1 - ((sum(s_Q_cal))/(sum(o_Q_cal))))
                
                # evaluate testing data
                yhat_test = model.predict(test_X_Q)
                test_X_Q = test_X_Q.reshape((test_X_Q.shape[0], nobs_Q))
                tmp_test = np.concatenate((test_X_Q, yhat_test), axis=1)
                inv_yhat_test = scaler.inverse_transform(tmp_test)
                NNytest = inv_yhat_test[:,-nsteps]
                test_X_Q = test_X_Q.reshape((test_X_Q.shape[0], nmonths, ninputs_Q))
                test_output_Q = np.array(NNytest)
                (s,o) = (test_output_Q[1:], yobs_test_Q[1:])
                NS_Q = 1 - sum((s-o)**2)/sum((o-np.mean(o))**2)
                PBIAS_Q = pc_bias(s,o)
                RMSE_Q = rmse(s,o)
                WB_Q = 1 - abs(1 - ((sum(s))/(sum(o))))
                
                # training and testing performances for Q
                perform = perform.append({'Dropout':  dropout_rate_value,'Hidden_units':hidden_unit_value,'Epochs':epochs_value , 'Batch_size_Q':batch_size_value,'NSE_cal_Q':NS_Q_cal,'PBIAS_cal_Q':PBIAS_Q_cal ,'RMSE_cal_Q':RMSE_Q_cal,'WB_cal_Q':WB_Q_cal,'NSE_val_Q':NS_Q,'PBIAS_val_Q':PBIAS_Q ,'RMSE_val_Q':RMSE_Q,'WB_val_Q':WB_Q },ignore_index = True)
                print(perform)
                
                output_df['Q_sim'] = test_output_Q
                
                # save model output in .csv
                output_df.to_csv('ML_LSTM_test_outputs.csv')


