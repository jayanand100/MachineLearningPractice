import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sb

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

#scale to get best performance

x_train = x_train/255
x_test = x_test/255

#flatten to one dimension

X_train_flattened = x_train.reshape(len(x_train), 28*28)
X_test_flattened = x_test.reshape(len(x_test), 28*28)

model = keras.Sequential([
    keras.layers.Dense(10, input_shape=(784,), activation='sigmoid')

])

model.compile(optimizer='adam',
              loss= 'sparse_categorical_crossentropy',
              metrics= ['accuracy'])
model.fit(X_train_flattened, y_train, epochs=5)

#predicted vs true target

print(model.evaluate(X_test_flattened, y_test))
y_predicted = model.predict(X_test_flattened)

#since every prediction gives a prediction of 10 values... assigning
#max probablity entry from the array

y_predicted_label = [np.argmax(i) for i in y_predicted]
#confusion matrix

cm = tf.math.confusion_matrix(labels=y_test, predictions=y_predicted_label)

plt.figure(figsize=(10,7))
sb.heatmap(cm, annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()