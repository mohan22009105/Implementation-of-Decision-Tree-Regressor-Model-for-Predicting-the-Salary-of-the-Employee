# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the -Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
1.Hardware – PCs
2,Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the libraries and read the data frame using pandas.

2.Calculate the null values present in the dataset and apply label encoder.

3.Determine test and training data set and apply decison tree regression in dataset.

4.Calculate Mean square error,data prediction and r2.

5.Print the obtained values.

## Program:

```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: arun.j
RegisterNumber:  212222040015
*/
import pandas as pd
data=pd.read_csv("/content/Salary.csv")
data.head()
data.info()
data.isnull().sum()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["Position"]=le.fit_transform(data["Position"])
data.head()
x=data[["Position","Level"]]
y=data["Salary"]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=2)
from sklearn.tree import DecisionTreeRegressor
dt=DecisionTreeRegressor()
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
from sklearn import metrics
mse=metrics.mean_squared_error(y_test,y_pred)
mse
r2=metrics.r2_score(y_test,y_pred)
print(r2)
dt.predict([[5,6]])
```

## Output:

## data.head()
![ML71](https://github.com/22009011/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/118343461/d8604749-6b07-489b-ae73-41103b4edf8a)

## data.info()
![ML72](https://github.com/22009011/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/118343461/6f6f973d-7df1-4565-9b36-554fca5ee2e4)

## isnull() and sum() function
![ML73](https://github.com/22009011/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/118343461/12632bec-e3cc-42d9-8728-11b00389f12f)

## data.head() for Position
![ML74](https://github.com/22009011/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/118343461/59a15106-f654-4e1b-b527-79efe8549e62)

## MSE value
![ML75](https://github.com/22009011/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/118343461/e1e2de6f-12c5-4968-8a48-0351494e5039)

## R2 value
![ML76](https://github.com/22009011/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/118343461/c5a0f972-6e83-49db-a27f-afad3763d876)

## Prediction value
![ML77](https://github.com/22009011/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/118343461/421470c6-6171-4a8e-bd3b-a2d420e4e094)

## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
