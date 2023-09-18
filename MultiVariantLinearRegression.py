import pandas as pd
from word2number import w2n
from sklearn import linear_model

df = pd.read_csv("C:\\Users\\annaj\\Downloads\\hiring.csv")
med = df['test_score(out of 10)'].median()
df['test_score(out of 10)'] = df['test_score(out of 10)'].fillna(med)

df['experience'] = df['experience'].fillna('zero')
df['experience'] = [w2n.word_to_num(x) for x in df['experience']]
med = df['experience'].median()
df['experience'] = df['experience'].replace(0, med)

reg = linear_model.LinearRegression()
reg.fit(df[['experience','test_score(out of 10)','interview_score(out of 10)']], df['salary($)'])
result = reg.predict([[0, 0, 1]])
print(reg.coef_)
print(reg.intercept_)
print(result)
print(df)







