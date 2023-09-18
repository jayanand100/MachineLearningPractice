import math
from sklearn import linear_model
import numpy as np
import pandas as pd

df = pd.read_csv("C:\\Users\\annaj\\Downloads\\test_scores.csv")
x = np.array(df.math)
y = np.array(df.cs)

def gradient_descent(x,y):
    m_curr = b_curr = 0
    iterations = 10000
    n = len(x)
    learning_rate = .0002
    cost_prev = 0

    for i in range(iterations):
        y_predicted = m_curr*x + b_curr
        cost = (1 / n) * sum([val ** 2 for val in (y - y_predicted)])

        md = -(2/n)*sum(x*(y-y_predicted))
        bd = -(2/n)*sum(y-y_predicted)

        m_curr = m_curr - learning_rate*md
        b_curr = b_curr - learning_rate*bd

        if math.isclose(cost, cost_prev, rel_tol=1e-20):
            break
        cost_prev = cost

        print(f"iteration = {i}; m = {m_curr}; b = {b_curr} and cost = {cost}")

gradient_descent(x,y)

reg = linear_model.LinearRegression()
reg.fit(df[['math']].values, df['cs'].values)
print(reg.coef_)
print(reg.intercept_)




