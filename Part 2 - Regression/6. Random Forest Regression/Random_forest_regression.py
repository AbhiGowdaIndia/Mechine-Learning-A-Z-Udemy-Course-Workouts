# Random forest Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
x = dataset.iloc[:,1:2].values
y = dataset.iloc[:,2].values

"""# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)"""

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train.reshape(-1,1))"""

#Fitting Linear Regression to the dataset
from sklearn.ensemble import RandomForestRegressor
regressor=RandomForestRegressor(n_estimators=300,random_state=0)
regressor.fit(x,y)

#Visualizing the Random forest regression results
x_grid=np.arange(min(x),max(x),0.01)
x_grid=x_grid.reshape((len(x_grid)),1)
plt.scatter(x,y,color="red")
plt.plot(x_grid,regressor.predict(x_grid),color="blue")
plt.title("Truth or Bluff (Polynomial Linear Regression)")
plt.xlabel("Position level")
plt.ylabel("Salary")
plt.show()

#predicting the new result with the linear regression
regressor.predict([[6.5]])


