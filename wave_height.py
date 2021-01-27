#!/usr/bin/env python
# _*_ coding: utf-8 _*_

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time

from wavesimulator_tensorflow import WaveSimulator
from sklearn.preprocessing import StandardScaler

# time the code
t0 = time.time()

#############################################################################
# Training and Validation using data in year 2005-2014 (10 years)
#############################################################################

# load the training/validation datasets
Inputs = pd.read_csv("inputs.csv", header=None)
Mask = pd.read_csv("mask.csv", header=None, names=np.arange(156,4049,1))
Hs = pd.read_csv("ml_Hs.csv", header=None)

# normalize Wind data
Features_Scaler = StandardScaler()
df_temp = Features_Scaler.fit_transform(Inputs)
Normalized_Inputs = pd.DataFrame(data=df_temp)

# join the Inputs and Mask Dataframes
Features = Normalized_Inputs.join(Mask)
t1 = time.time()
print("Data loading and processing take {:.5f} seconds".format(t1-t0))

# create the wave height simulator
WhMLP = WaveSimulator(Features, Hs)
WhMLP.fit_model()
WhMLP.model_predict()
t2 = time.time()
print("MLP model training and validation take {:.5f} seconds".format(t2-t1))

# compare the results
y_pred = pd.DataFrame(WhMLP.y_pred)
y_test_values = WhMLP.y_test.values
y_data = pd.DataFrame(y_test_values)

# compare domain-averaged values
True_values_mean = y_data.mean(axis=1)
Predict_values_mean = y_pred.mean(axis=1)

True_values_mean[True_values_mean<0] = 0
Predict_values_mean[Predict_values_mean<0] = 0
# Write results to files
#True_values_mean.to_csv(path='TrueMean.csv', sep=" ", header=None)   # save results to files
#Predict_values_mean.to_csv(path='PredictMean.csv', sep=" ", header=None)

plt.figure()
# plt.plot(True_values_mean,'r-')
# plt.plot(Predict_values_mean,'b-')
# plt.xlabel("True mean Hs (m)",fontsize=14)
# plt.ylabel("Predicted mean Hs (m)",fontsize=14)
# plt.show()

sns.jointplot(True_values_mean,Predict_values_mean,kind="reg")
plt.xlabel("True mean Hs (m)",fontsize=14)
plt.ylabel("Predicted mean Hs (m)",fontsize=14)
plt.subplots_adjust(0.12,0.12)
plt.show()

#####################################################################################
# Testing in year 2015
#####################################################################################
Inputs15 = pd.read_csv("inputs2015.csv", header=None)
Mask15 = pd.read_csv("mask2015.csv", header=None, names=np.arange(156,4049,1))
Hs15 = pd.read_csv("Hs2015.csv", header=None)

df15 = Features_Scaler.fit_transform(Inputs15)
Inputs15 = pd.DataFrame(data=df15)
X_fore = Inputs15.join(Mask15)

t3 = time.time()
y_fore = WhMLP.wave_forecast(X_fore)
y_fore = pd.DataFrame(y_fore)
t4 = time.time()
print("Wave height forecasting takes {:.5f} seconds".format(t4-t3))

Measured_values_mean = Hs15.mean(axis=1)
Forecast_values_mean = y_fore.mean(axis=1)
# Write results to files
#Measured_values_mean.to_csv(path='Hs15Mean.csv', sep=" ", header=None)
#Forecast_values_mean.to_csv(path='ForeMean.csv', sep=" ", header=None)

tim = np.linspace(0,365,1460)

plt.figure()
plt.plot(tim,Measured_values_mean,'r--')
plt.plot(tim,Forecast_values_mean,'b-')
plt.xlabel("Days from 2015-01-01",fontsize=14)
plt.ylabel("Spatially-averaged Hs (m)",fontsize=14)
plt.show()

sns.jointplot(Measured_values_mean,Forecast_values_mean,kind="reg")
plt.xlabel("True mean Hs (m)",fontsize=14)
plt.ylabel("Predicted mean Hs (m)",fontsize=14)
plt.subplots_adjust(0.15,0.12)
plt.show()

score = WhMLP.performance_metric(Hs15, y_fore)
print("{} model has an prediction R2 score: {:.2f}".format(WhMLP.model, score))