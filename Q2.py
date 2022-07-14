'''NAME : PUNISH KUMAR
   ROLL NO. : B20308
   MOB.NO : 8882286890'''
import pandas as pd
import numpy as np
from statsmodels.tsa.ar_model import AutoReg as AR
import math
import matplotlib.pyplot as plt


data = pd.read_csv('C:/Users/Manish/Downloads/daily_covid_cases.csv',parse_dates=['Date'],index_col=['Date'],sep=',')

#a
test_size = 0.35
X = data.values

tst_siz = math.ceil(len(X)*test_size)
train,test = X[:len(X)-tst_siz],X[len(X)-tst_siz:]

#AR model
p=5
model = AR(train,lags=p)
model_fit = model.fit()   #fit train model
coef = model_fit.params   
r_coef = np.round(coef,3)
print(f'Coefficients of AR model are {r_coef}')
history = train[len(train)-p:]
history = [history[i] for i in range(len(history))]
predictions =[]

for i in range(len(test)):
    length = len(history)
    lag = [history[j] for j in range(length-p,length)]
    y_hat= coef[0]

    for d in range(p):
        y_hat += coef[d+1]*lag[p-d-1]    
    predictions.append(y_hat)
    obs = test[i]
    history.append(obs)


#i)
plt.scatter(test,predictions)
plt.xlabel('actual test data')
plt.ylabel('predicted data')
plt.title('scatter plot betwwen actual and predicted values')
plt.show()

#ii)
plt.plot(test,predictions)
plt.xlabel('actual test data')
plt.ylabel('predicted data')
plt.title('line plot betwwen actual and predicted values')
plt.show()

#iii)
n= len(test)
e = 0
for i in range(len(test)):
    e += (predictions[i]-test[i])**2      
RMSE = math.sqrt(e/n)
RMSE_per = (RMSE/test.mean())*100
print(f'RMSE % is {RMSE_per}')

#MAPE
s = 0
for i in range(len(test)):
    s += float(abs(test[i]-predictions[i])/test[i])
MAPE = (s/n)*100             
print(f'MAPE between actual and predicted data is {MAPE}')


