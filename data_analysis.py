import pandas as pd
import numpy as np
import histogram as hist
import matplotlib.pyplot as plt
import dataprocess as dp
from sklearn import preprocessing
import matplotlib.dates as mdate
from sklearn import neighbors

dataset = pd.read_excel('F:\Python projects\gas-dataset-regression\AirQualityUCI\AirQualityUCI.xlsx')
dataset['DT'] = pd.to_datetime(dataset['DT'])
dataset = dataset.set_index('DT')
dates = dataset.index
featureset = ['DT','PT08.S1(CO)', 'PT08.S2(NMHC)','T', 'PT08.S3(NOx)', 'PT08.S4(NO2)', 'RH']
target = ['DT', 'CO(GT)']
knn = neighbors.KNeighborsClassifier()

'''
rh_distribution = hist.Kde(rh)
anim = rh_distribution.getKdeAni()
'''
'''
data_piece = dataset['2004-12-01':'2005-02-28']
dates = data_piece.index
data = data_piece[['RH', 'CO(GT)']]
data = data.replace(-200.00, np.nan)
RH , CO= data['RH'], data['CO(GT)']
fig = plt.figure()
fig.gca().xaxis.set_major_formatter(mdate.DateFormatter('%Y-%m-%d %H:%M:%S'))
fig.gca().xaxis.set_major_locator(mdate.WeekdayLocator())
ax1 = fig.add_subplot(111)
ax1.plot(dates.values, CO.values, color = 'red', lw = 0.5)
ax1.set_ylabel('Concentration of CO')
ax2 = ax1.twinx() # this is the important function
ax2.set_ylabel('Relative Humidity')
ax2.plot(dates.values, RH.values, color = 'blue' ,lw = 0.5)
plt.gcf().autofmt_xdate()
plt.show()
'''
y1 = [0.0201, 0.0220, 0.0304, 0.0336, 0.0377, 0.0397, 0.0400, 0.0392]  #results of C6H6
y2 = [151.5, 166.5, 195.9, 226.4, 247.8, 231.8, 216.5, 197.0]   #results of CO
m = ['2004-3', '2004-4', '2004-5','2004-6','2004-7', '2004-8', '2004-9', '2004-10', '2004-11', '2004-12', '2005-1', '2005-2', '2005-3']
m = pd.to_datetime(m)  #dates
x = [15, 25, 35, 45, 55, 65, 75, 85]  #rh
'''
fig = plt.figure()
#fig.gca().xaxis.set_major_formatter(mdate.DateFormatter('%Y-%m'))
#fig.gca().xaxis.set_major_locator(mdate.MonthLocator())
ax1 = fig.add_subplot(111)
ax1.plot(x, y1, color = 'blue', lw =1, label = 'C6H6')
ax1.scatter(x,y1,color = 'blue')
ax1.set_ylabel('MAE(C6H6)')
ax1.set_xlabel('RH')
ax2 = ax1.twinx() # this is the important function
ax2.plot(x, y2, color = 'red', lw =1, label = 'CO')
ax2.scatter(x,y2,color = 'red')
ax2.set_ylabel('MAE(CO)')
ax2.set_xlabel('RH')
legend1 = ax1.legend(loc=(.02, .8), fontsize=12)
legend2 = ax2.legend(loc=(.02, .9), fontsize=12)
plt.xlim(0,100)
plt.gcf().autofmt_xdate()
plt.show()
'''
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.plot(x, y2, color = 'blue', lw =1)
ax1.scatter(x,y2,color = 'blue')
ax1.set_ylabel('std_deviation of Sensor')
ax1.set_xlabel('RH')
plt.xlim(0,100)
plt.gcf().autofmt_xdate()
plt.show()