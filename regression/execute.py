import pandas as pd
import quandl, math, datetime
import numpy as np
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style
import pickle # save some time for not retraining algorithm every time.
style.use('ggplot') #how plot is going to be looked like

dataframe = quandl.get('WIKI/GOOGL')
#print(dataframe.head())

#Grabbing features from dataframe
dataframe = dataframe[['Adj. Open','Adj. High','Adj. Low','Adj. Close','Adj. Volume',]]
#Define percent volatility between low-high for each day
dataframe['High/Low Perc.'] = (dataframe['Adj. High'] - dataframe['Adj. Low']) / dataframe['Adj. Low'] * 100.0

#Define percent volatility between old-new for each day
dataframe['New/Old Perc.'] = (dataframe['Adj. Close'] - dataframe['Adj. Open']) / dataframe['Adj. Open'] * 100.0

dataframe = dataframe[['Adj. Open', 'Adj. Close', 'New/Old Perc.', 'Adj. High', 'Adj. Low', 'High/Low Perc.', 'Adj. Volume', ]]


forecast_col = 'Adj. Close'
dataframe.fillna(value=-99999, inplace=True) # pandas term, Nan data, therefore we are going to replace Nan data with some extreme numbers so that we can see the boundaries of our model

forecast_out = int(math.ceil(0.01 * len(dataframe))) #to the nearest number. How many days do we want to take care of instead of forecasting today or tomorrows Adj. Close.
                                                    # we can do it by changing the value of 0.1
print(forecast_out)

dataframe['label'] = dataframe[forecast_col].shift(-forecast_out) #shifting the days based on the day value of forecast_out

#print(dataframe.head())

X = np.array(dataframe.drop(['label'], 1)) # dataframe drop function is returning a new dataframe.
                                           # we want to return all the attributes of dataframe except the label attribute
                                           # to be a somehow lazy :)

# now scaling X.
# We scaling X before we feed it though the classifier but let's say we feed it through class far we have a classifier and
# then we are using it in real time on real data.
# scaling enables it all scaled together so it's like normalized with all the other data points so in order to propoerly scale it
# you would have to include it with your training data so keep that in mind if you ever go into the future and you are actually
# using below method, you need to scale new values but not just scale them, but scale own alongside all your other values.
# it can help training testing, it can actually add processing time especially if you're doing like doing high frequency trading.
# you weould always skip that step.
X = preprocessing.scale(X)
dataframe.dropna(inplace=True)

X_lately = X[-forecast_out:]

# now redefining x
# X is being equalt to X to the point of where we were able to forecast
# we make sure that we only have X;s where we have values for y
X = X[:-forecast_out:]


Y = np.array(dataframe['label'])

#print(len(X), len(Y))

X_train, X_test, Y_train, Y_test = cross_validation.train_test_split(X, Y, test_size=0.2)
#Linear regression algorithm
#it runs linearly or 1 process as a default
#classifier = LinearRegression()
#we can define n_jobs to do parallel processing. To speed up training process.
#   classifier = LinearRegression(n_jobs=10)
# or you can set n_jobs to -1 which lead it to run as many process as possible by my processor
classifier = LinearRegression(n_jobs=-1)
classifier.fit(X_train, Y_train)
# -- Saving classifier for preventing training every time
# -- like saving models
# with open('linearregression.pickle', 'wb') as f:
#     pickle.dump(classifier,f)
pickle_in = open('linearregression.pickle', 'rb')
classifier = pickle.load(pickle_in)
accuracy = classifier.score(X_test, Y_test)
print(accuracy)

#support vector regression classifier algoritm
classifier_svr = svm.SVR()
classifier_svr.fit(X_train, Y_train)
accuracy_svr = classifier_svr.score(X_test, Y_test)
print(accuracy_svr)

#support vector regression classifier algoritm with specifying Kernel
classifier_svr = svm.SVR(kernel='poly')
classifier_svr.fit(X_train, Y_train)
accuracy_svr = classifier_svr.score(X_test, Y_test)
print(accuracy_svr)

#print(accuracy)

#next 32 days stock prices
forecast_set = classifier.predict(X_lately)
print(forecast_set, accuracy, forecast_set )

dataframe['Forecast'] = np.nan
last_date = dataframe.iloc[-1].name
last_unix = last_date.timestamp()
one_day = 86400
next_unix= last_unix + one_day

# populate dataframe with the new dates and forecasted values
# iterating through the forecast set taking each forecast and day
# and then setting those as the values in the dataframe basically
# making the future features not a number
# and the last line just takes all of the first columns set tem to
# not a numbers and the final column is whatever I is which is the forecast in this case

for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += one_day
    dataframe.loc[next_date] = [np.nan for _ in range (len(dataframe.columns) - 1)] + [i]

dataframe['Adj. Close'].plot()
dataframe['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()
































