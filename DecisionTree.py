import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv("C:\\Users\\annaj\Downloads\\titanic.csv")
df.drop(['PassengerId','Name','SibSp','Parch','Ticket','Cabin','Embarked'],axis='columns',inplace=True)
inputs = df.drop('Survived', axis='columns')
target = df.Survived

inputs['Sex'] = inputs['Sex'].map({'male':1, 'female':2})
inputs['Age'] = inputs['Age'].fillna(inputs.Age.mean())

x_train, x_test, y_train, y_test = train_test_split(inputs,target, train_size=.8)

model = DecisionTreeClassifier()
model.fit(x_train, y_train)

print(model.score(x_test,y_test))