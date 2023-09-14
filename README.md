# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.import pandas module.

2.Read the required csv file using pandas .

3.Import LabEncoder module.

4.From sklearn import logistic regression.

5.Predict the values of array.

6.Calculate the accuracy, confusion and classification report by importing the required modules from sklearn.

7.print the required values.

8.End the program.

## Program:
```

Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
### Developed by: PRANEET S
### RegisterNumber:  212221230078


import pandas as pd
data = pd.read_csv("Placement_Data.csv")
print(data.head())
data1 = data.copy()
data1= data1.drop(["sl_no","salary"],axis=1)
print(data1.head())
data1.isnull().sum()
data1.duplicated().sum()
from sklearn.preprocessing import LabelEncoder
lc = LabelEncoder()
data1["gender"] = lc.fit_transform(data1["gender"])
data1["ssc_b"] = lc.fit_transform(data1["ssc_b"])
data1["hsc_b"] = lc.fit_transform(data1["hsc_b"])
data1["hsc_s"] = lc.fit_transform(data1["hsc_s"])
data1["degree_t"]=lc.fit_transform(data["degree_t"])
data1["workex"] = lc.fit_transform(data1["workex"])
data1["specialisation"] = lc.fit_transform(data1["specialisation"])
data1["status"]=lc.fit_transform(data1["status"])
print(data1)
x = data1.iloc[:,:-1]
print(x)
y = data1["status"]
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(solver="liblinear")
print(lr.fit(x_train,y_train))
y_pred = lr.predict(x_test)
print(y_pred)
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test,y_pred)
print(accuracy)
from sklearn.metrics import confusion_matrix
confusion = confusion_matrix(y_test,y_pred)
print(confusion)
from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,y_pred)
print(classification_report1)
print(lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]]))

```

## Output:

![ML41](https://github.com/BHARATHWAJRAMESH/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119394248/20416035-1543-499d-a79e-6d1687f2c8fe)

![ML42](https://github.com/BHARATHWAJRAMESH/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119394248/76b8821b-7c94-49b9-b401-71e477bdaaa8)

![ML43](https://github.com/BHARATHWAJRAMESH/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119394248/2c6fb62a-4295-4cd9-bd1a-3a1025c30b1f)

![ML44](https://github.com/BHARATHWAJRAMESH/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119394248/ad8e3828-8361-4fe3-8d8b-2d6f1f8c591b)

![ML45](https://github.com/BHARATHWAJRAMESH/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119394248/394c158d-8ec3-47a8-9396-75c01462f769)

![ML46](https://github.com/BHARATHWAJRAMESH/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119394248/4774d054-0886-4d3a-8a34-3b29af9ad67e)

![ML47](https://github.com/BHARATHWAJRAMESH/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119394248/4d83c15a-4109-4768-88c3-db290e3c0ae6)

![ML48](https://github.com/BHARATHWAJRAMESH/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119394248/eb58677e-66a4-4628-a0fb-c27fd6d18f47)


## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
