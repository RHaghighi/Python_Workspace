#! /usr/bin/python

import csv
import pandas as pd
import math, datetime
import numpy as np
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style
import pickle

style.use('ggplot')

'''
with open('WIKI_GOOGL.csv') as csvfile:
	reader = csv.reader(csvfile)
	for row in reader:
		print(row)
'''


df = pd.read_csv('WIKI_GOOGL.csv', index_col=0, parse_dates=True)

print(df.head())

df = df[['adj_open','adj_high','adj_low','adj_close','adj_volume',]]
df['HL_PCT'] = (df['adj_high']-df['adj_close']) / df['adj_close'] * 100.0
df['PCT_change'] = (df['adj_close']-df['adj_open']) / df['adj_open'] * 100.0

df = df[['adj_close','HL_PCT','PCT_change','adj_volume']]  

#print(df.head())

forecast_col = 'adj_close'
df.fillna(-99999, inplace=True)

forecast_out = int(math.ceil(0.01*len(df)))

print(forecast_out)

df['label']=df[forecast_col].shift(-forecast_out)

df.dropna(inplace=True)
print(df.head())

X = np.array(df.drop(['label'],1))
X = preprocessing.scale(X)

X_lately = X[-forecast_out:]
X = X[:-forecast_out]

df.dropna(inplace=True)
y = np.array(df['label'])
y = y[:-forecast_out]
print(len(X),len(y))

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)

clf = LinearRegression(n_jobs=-1)
#clf = svm.SVR(kernel='rbf')
clf.fit(X_train, y_train)

with open('linearregression.pickle','wb') as f:
	pickle.dump(clf, f)

pickle_in = open('linearregression.pickle','rb')
clf = pickle.load(pickle_in)


accuracy = clf.score(X_test,y_test)

print(accuracy)

forecast_set = clf.predict(X_lately)

print(forecast_set, accuracy, forecast_out)

df['Forecast'] = np.nan

last_date = df.iloc[-1].name
#print(last_date)
last_unix = last_date.value / 1e9
#print(last_unix)
one_day = 86400
print(datetime.datetime.fromtimestamp(last_unix))
next_unix = last_unix + one_day
print(next_unix)

for i in forecast_set:
	next_date = datetime.datetime.fromtimestamp(next_unix)
	next_unix += one_day
	df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)]+[i]


df['adj_close'].plot()
df['Forecast'].plot()
plt.xlabel('Date')
plt.ylabel('price')
plt.show()
