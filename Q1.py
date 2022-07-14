

""" Name- Punish Kumar
    Roll No - B20308
    Mob no - 8882286890 """
# Python code for Question-1 

# import python libraries
import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.dates as mdates
import statsmodels.api as sm
from matplotlib.dates import DateFormatter
from statsmodels.graphics.tsaplots import plot_acf

# read the csv file using pandas library
data=pd.read_csv("daily_covid_cases.csv")

# store the original new cases data in the old_data variable
old_data=data['new_cases']

# Part-a
# plot the line of new covid cases v/s date
# set the axix value title and label with ax function
fig, ax = plt.subplots(figsize=(12,6))
ax.plot(data['Date'],data['new_cases'].values,color='red')
ax.set(xlabel="Date", ylabel="New_Covid_Cases",title="Date v/s new_Covid_cases Line Plot")
form_date = DateFormatter("%b-%d")
ax.xaxis.set_major_formatter(form_date)
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
plt.xticks(rotation =90)
plt.show()

# Part-b
# find the Pearson correlation (autocorrelation) coefficient between orignal data and one day lagged data using corr function
print("Part-B ---> ")
lag1_data=old_data.shift(1)
print("Pearson correlation (autocorrelation) coefficient is: ",old_data.corr(lag1_data))
print("-------")

# Part-c
# Scatter plot of the orignal time series data and one day lagged data
plt.scatter(old_data, lag1_data, s=5, color="red")
plt.xlabel("Given time sequence data")
plt.ylabel("One day lagged generated time sequence data")
plt.title("Scatter plot")
plt.show()

# Part-d
# find the time lagged data series and it's correlation cofficient with original data
# plot the line graph between lag time and it's correlation cofficient
PCC=sm.tsa.acf(old_data)
lag_time=[1,2,3,4,5,6]
pcc=PCC[1:7]
plt.plot(lag_time,pcc, marker='o')
for xitem,yitem in np.nditer([lag_time, pcc]):
        etiqueta = "{:.3f}".format(yitem)
        plt.annotate(etiqueta, (xitem,yitem), textcoords="offset points",xytext=(0,10),ha="center")
plt.xlabel("Time Lag value")
plt.ylabel("Correlation coffecient value")
plt.title("line plot btw obtained correlation coefficients and lagged values")
plt.show()

# Part-e
# Plot Auto Correlation Function using python inbuilt function plot_acf
# plot the line diagram of correlation cofficient with respect to increase the time lag value
plot_acf(x=old_data, lags=50, color="purple")
plt.xlabel("Lag value")
plt.ylabel("Correlation coffecient value")
plt.title("Line plot btw Lag value and Correlation coffecient value")
plt.show()