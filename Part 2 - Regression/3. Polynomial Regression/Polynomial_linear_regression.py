# polynomial linear regression

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
from sklearn.linear_model import LinearRegression
lin_reg=LinearRegression()
lin_reg.fit(x,y)

from sklearn.preprocessing import PolynomialFeatures
poly_reg=PolynomialFeatures(degree=4)
x_poly=poly_reg.fit_transform(x)
lin_reg_2=LinearRegression()
lin_reg_2.fit(x_poly,y)

#Visualizing the linear regression results
plt.scatter(x,y,color="red")
plt.plot(x,lin_reg.predict(x),color="blue")
plt.title("Truth or Bluff (Linear Regression)")
plt.xlabel("Position level")
plt.ylabel("Salary")
plt.show()

#Visualizing the polynomial regression results
x_grid=np.arange(min(x),max(x),0.1)
x_grid=x_grid.reshape((len(x_grid)),1)
plt.scatter(x,y,color="red")
plt.plot(x,lin_reg_2.predict(poly_reg.fit_transform(x)),color="blue")
plt.title("Truth or Bluff (Polynomial Linear Regression)")
plt.xlabel("Position level")
plt.ylabel("Salary")
plt.show()

#predicting the new result with the linear regression
lin_reg.predict([[6.5]])


#Predicting the new result with the Ploynomial lnear regression
lin_reg_2.predict(poly_reg.fit_transform([[6.5]]))