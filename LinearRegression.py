import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model

df = pd.read_csv("C:\\Users\\annaj\\Downloads\\canada_per_capita_income.csv")

plt.xlabel("year")
plt.ylabel("pci(US$)")
plt.scatter(df.year, df['per capita income (US$)'], color= 'red', marker='+')
plt.show()

reg = linear_model.LinearRegression()
reg.fit(df[['year']].values, df['per capita income (US$)'].values)

plt.xlabel("year")
plt.ylabel("pci(US$)")
plt.scatter(df[['year']].values, df['per capita income (US$)'].values, color= 'red', marker='+')
plt.plot(df['year'], reg.predict(df[['year']].values), color='blue')
plt.show()
