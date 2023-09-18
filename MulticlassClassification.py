from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import seaborn as sb
import matplotlib.pyplot as plt

iris_ds = load_iris()
print(dir(iris_ds))

print(iris_ds.feature_names)
print(len(iris_ds.target))
print(iris_ds.target_names)


x_train, x_test, y_train, y_test = train_test_split(iris_ds.data, iris_ds.target, train_size=.7, random_state=1)

model = LogisticRegression()
print(model.fit(x_train, y_train))

print(model.score(x_test,y_test))
y_pred = model.predict(x_test)

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(10,7))
sb.heatmap(cm, annot=True)
plt.xlabel('pred')
plt.ylabel('truth')
plt.show()