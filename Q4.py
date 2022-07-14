#Name - PUNISH KUMAR
#Roll No - B20308
#Contact No - 8882286890

#importing required libraries
import pandas as pd
import math
import numpy as np
from statsmodels.tsa.ar_model import AutoReg as AR

data = pd.read_csv('daily_covid_cases.csv',parse_dates=['Date'],index_col=['Date'],sep=',')

test_size = 0.35
X = data.values
tst_sz = math.ceil(len(X)*test_size)
train, test = X[:len(X)-tst_sz], X[len(X)-tst_sz:]

test1 = pd.DataFrame(test,columns =['orig'])

cor=[]
b = (2/math.sqrt(len(train)))
print('Correlation coefficient:')
for i in range(1,101):
    name=f'lag{i}'
    data[name]=data['new_cases'].shift(i)
    a = np.abs(data['new_cases'].corr(data[name]))
    cor.append(a)
    
for i in range(100):
    if cor[i] > b:
        print(f'````` For {i+1}-day lag :',cor[i])
    else:
        print(f'@@@@@ For {i+1}-day lag :',cor[i])
  

p=47
model = AR(train, lags=p, old_names=False)
model_fit = model.fit()
coef = model_fit.params

hist = train[len(train)-p:]
hist = [hist[i] for i in range(len(hist))]

pred = []
for i in range(len(test)):
    y = coef[0]
    for t in range(p):
        y += coef[t+1] * hist[p-t-1]

    pred.append(y)
    hist.append(test[i])
    hist.pop(0)

n=0
m=0
d=0
for i in range(len(test)):
    n += (test[i]-pred[i])**2
    d += test[i]
    f = (test[i]-pred[i])/test[i]
    if f > 0:
        m += f
    else:
        m = m - f
n = (n/len(test))**0.5
d = d/len(test)
n = float(n/d)*100
m = float(m/len(test))*100

print('')
print(f'For lag = {p},\n RMSE(%) = {round(n,3)} \n MAPE = {round(m,3)}')

