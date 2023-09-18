#this is done when the relation is a SIGMOID/LOGIT function
from sklearn import linear_model, model_selection
import pandas as pd

df = pd.read_csv("C:\\Users\\annaj\\Downloads\\HR_comma_sep.csv")

subdf = df[['satisfaction_level','average_montly_hours','promotion_last_5years','salary']]

salary_dummies = pd.get_dummies(subdf.salary, prefix='salary' )
df_with_dummies = pd.concat([subdf,salary_dummies], axis='columns')
df_with_dummies.drop('salary', axis='columns', inplace=True)

x = df_with_dummies
y = df.left

x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, train_size=.3)

model = linear_model.LogisticRegression()
model.fit(x_train,y_train)

print(model.predict(x_test))
print(model.score(x_test, y_test))
