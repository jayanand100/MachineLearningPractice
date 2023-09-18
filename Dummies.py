import pandas as pd
from sklearn import linear_model

df = pd.read_csv("C:\\Users\\annaj\\Downloads\\carprices.csv")


#using pandas dummies when nominal variables

a = pd.get_dummies(df['Car Model'])
merged = pd.concat([df,a], axis='columns')
b = merged.drop(['Car Model', 'Mercedez Benz C class'],axis='columns')

X = b.drop(['Sell Price($)'], axis='columns')

reg = linear_model.LinearRegression()
reg.fit(X,df['Sell Price($)'])
print(reg.coef_)
print(reg.intercept_)


#Also can use one hot encoding when nominal variables from sklearn






