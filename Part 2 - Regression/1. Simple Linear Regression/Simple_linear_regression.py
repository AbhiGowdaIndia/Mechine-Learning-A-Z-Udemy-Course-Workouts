# Simple linear regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Salary_Data.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:,1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 1/3, random_state = 0)

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train.reshape(-1,1))"""

#Fitting simple linear regression to the training set
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)

#predicting the test set result
y_pred=regressor.predict(x_test)

##Visualizing the training set results
#plt.scatter(x_train,y_train,color='r')
#plt.plot(x_train,regressor.predict(x_train),color='b')
#plt.title("Salary vs Experience (training set)")
#plt.xlabel("Experience")
#plt.ylabel("Salary")
#plt.show()
#
##Visualizing the test set results
#plt.scatter(x_test,y_test,color='g')
#plt.plot(x_train,regressor.predict(x_train),color='b')
#plt.title("Salary vs Experience (test set)")
#plt.xlabel("Experience")
#plt.ylabel("Salary")
#plt.show()
#
##Visualizing the model predicted results
#plt.scatter(x_test,y_pred,color='k')
#plt.plot(x_train,regressor.predict(x_train),color='b')
#plt.title("Salary vs Experience (predicted)")
#plt.xlabel("Experience")
#plt.ylabel("Salary")
#plt.show()

#Visualizing training set, test set and  prdicted values
plt.scatter(x_train,y_train,color='r')
plt.scatter(x_test,y_test,color='g')
plt.scatter(x_test,y_pred,color='k')
plt.plot(x_train,regressor.predict(x_train),color='b')
plt.title("Salary vs Experience")
plt.xlabel("Experience")
plt.ylabel("Salary")
plt.show()