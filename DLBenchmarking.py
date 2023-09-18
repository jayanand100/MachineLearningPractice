import tensorflow as tf
from tensorflow import keras
import time
import numpy as np
import matplotlib.pyplot as plt

fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

train_images = train_images/ 255.0
test_images = test_images / 255.0

train_labels_categorical = keras.utils.to_categorical(train_labels, num_classes= 10, dtype='float32')
test_labels_categorical = keras.utils.to_categorical(test_labels, num_classes= 10, dtype='float32')


def get_model():
    model = keras.Sequential([
            keras.layers.Flatten(input_shape=(28,28)),
            keras.layers.Dense(3000, activation= 'relu'),
            keras.layers.Dense(1000, activation='relu'),
            keras.layers.Dense(10, activation='sigmoid')
        ])

    model.compile(
        optimizer= 'SGD',
        loss= 'categorical_crossentropy',
        metrics= ['accuracy']
    )

    return model

#140912.709ms with cpu
with tf.device('/CPU:0'):
    start = time.time()
    cpu_model = get_model()
    cpu_model.fit(train_images, train_labels_categorical, epochs=5)
    end = time.time()
    print(f"Time taken: {(end-start)*10**3:.03f}ms")




