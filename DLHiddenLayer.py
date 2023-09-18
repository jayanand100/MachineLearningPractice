import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sb

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

#scale to get best performance

x_train = x_train/255
x_test = x_test/255

#flatten to one dimension in keras

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),
    keras.layers.Dense(100, activation='relu'),
    keras.layers.Dense(10, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss= 'sparse_categorical_crossentropy',
              metrics= ['accuracy'])
model.fit(x_train, y_train, epochs=10)

#predicted vs true target

print(model.evaluate(x_test, y_test))
y_predicted = model.predict(x_test)

#since every prediction gives a prediction of 10 values... assigning
#max probable entry from the array

y_predicted_label = [np.argmax(i) for i in y_predicted]
#confusion matrix

cm = tf.math.confusion_matrix(labels=y_test, predictions=y_predicted_label)

plt.figure(figsize=(10,7))
sb.heatmap(cm, annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()