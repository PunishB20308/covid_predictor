
""" Name- Punish Kumar
    Roll No - B20308
    Mob no - 8882286890 """
# Python code for Question-3 (Lab-6)

# import python libraries
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
from statsmodels.tsa.ar_model import AutoReg as AR

# spilting the covid cases data into train (65%) and test data (35%)
series = pd.read_csv('daily_covid_cases.csv',parse_dates=['Date'],index_col=['Date'],sep=',')
train_size = 0.65
X = series.values
train, test = X[:int(len(X)*train_size)], X[int(len(X)*train_size):]

# define a AR mode for different test data, train data and lag value
def ARmodel(train_data, test_data, lag):
    window=lag
    model = AR(train_data, lags=window)
    # fit/train the model
    model_fit = model.fit()
    # Get the coefficients of AR model (w1,w2,w3......)
    coef = model_fit.params
    #using these coefficients walk forward over time steps in test, one step each time
    history = train_data[len(train_data)-window:]
    history = [history[i] for i in range(len(history))]
    # using these coefficients walk forward over time steps in test, one step each time
    predictions = list() 
    for t in range(len(test_data)):
        length = len(history)
        lag = [history[i] for i in range(length-window,length)]
        yhat = coef[0] 
        for d in range(window):
            yhat += coef[d+1] * lag[window-d-1]
        obs = test_data[t]
        # Append predictions to compute RMSE later
        predictions.append(yhat)
        # Append actual test value to history, to be used in next step
        history.append(obs)
    # find the rmse and mape value for different test data
    rmse_val=mean_squared_error(test_data, predictions,squared=False)*100/(sum(test_data)/len(test_data))
    mape_val=mean_absolute_percentage_error(test_data, predictions)
    return rmse_val, mape_val

# store the given lag value in a list 
# create empty list to store RMSE and MAPE value
lag=[1,5,10,15,25]
rmse_li=[]
mape_li=[]
for i in lag:
    rmse,mape=ARmodel(train, test,i)
    print(f"RMSE value for lag = {i} is : ", rmse)
    print(f"MAPE value for lag = {i} is : ", mape)
    rmse_li.append(rmse[0])
    mape_li.append(mape)

# plot the Bar chart between RMSE error and Lag values
plt.bar(lag, rmse_li,color="green")
plt.ylabel('RMSE error value')
plt.xlabel('Lag values')
plt.title("Bar chart between RMSE error and Lag values")
plt.xticks(lag)
plt.show()

# plot the Bar chart between MAPE error and Lag values
plt.bar(lag, mape_li,color="red")
plt.ylabel('MAPE error')
plt.xlabel('Lag values')
plt.title("Bar chart between MAPE error and Lag values")
plt.xticks(lag)
plt.show()