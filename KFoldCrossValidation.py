from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import numpy as np

iris = load_iris()

lr_scores = cross_val_score(LogisticRegression(), iris.data, iris.target)
dt_scores = cross_val_score(DecisionTreeClassifier(), iris.data, iris.target)
svc_score = cross_val_score(SVC(), iris.data, iris.target)
rf_score = cross_val_score(RandomForestClassifier(), iris.data, iris.target)

print(lr_scores)
print(np.average(lr_scores))

print(dt_scores)
print(np.average(dt_scores))

print(svc_score)
print(np.average(svc_score))

print(rf_score)
print(np.average(rf_score))  

