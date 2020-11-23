# -*- coding: utf-8 -*-
"""
Created on Mon Nov 16 20:03:42 2020

@author: Keerthana Kumaran
"""

import pandas as pd
import matplotlib.pyplot as plt
df=pd.read_csv('insurance.csv')
#matrix of X features
X=df.iloc[:,:-1].values
#vector of prediction values
y=df.iloc[:,-1].values
from sklearn.model_selection  import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=0)

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train[:,:]=sc.fit_transform(X_train[:,:])
X_test[:,:]=sc.fit_transform(X_test[:,:])

y_train=y_train.reshape(-1,1)
y_test=y_test.reshape(-1,1)

#y_train[:]=sc.fit_transform(y_train[:])
#y_test[:]=sc.fit_transform(y_test[:])

from sklearn.linear_model import  LinearRegression
reg=LinearRegression()
reg.fit(X_train,y_train)



y_pred=reg.predict(X_test)
y_pred=y_pred.reshape(-1,1)
print(reg.score(X_test,y_test))
print("reg coefficents:")
print(reg.coef_)
print(reg.intercept_)