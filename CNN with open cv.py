import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import os
import tensorflow as tf

from tensorflow import keras
from keras import layers
from keras.models import Sequential

import pathlib


'''dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"

data_dir = tf.keras.utils.get_file('flower_photos', origin= dataset_url, cache_dir='.', untar=True)'''

data_dir = "./datasets/flower_photos"

data_dir = pathlib.Path(data_dir)

'''roses = list(data_dir.glob('roses/*'))

plt.imshow(mpimg.imread(roses[1]))
plt.show()''' #this displays the image


flowers_images_dict = {
    'roses': list(data_dir.glob('roses/*')),
    'daisy': list(data_dir.glob('daisy/*')),
    'dandelion': list(data_dir.glob('dandelion/*')),
    'sunflowers': list(data_dir.glob('sunflowers/*')),
    'tulips': list(data_dir.glob('tulips/*')),
}


flowers_labels_dict = {
    'roses': 0,
    'daisy': 1,
    'dandelion': 2,
    'sunflowers': 3,
    'tulips': 4,
}

X, y = [], []

img_height, img_weight = 100, 100

for flower_name, images in flowers_images_dict.items():
    for image in images:
        img = cv2.imread(str(image))
        resized_image = cv2.resize(img, (img_height, img_weight))
        X.append(resized_image)
        y.append(flowers_labels_dict[flower_name])

X = np.array(X)
y = np.array(y)

#train test split

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

X_train_scaled = X_train / 255
X_test_scaled = X_test / 255

num_classes = 5

model = Sequential([
    layers.Conv2D(16, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes)

])
 #linux specific...
 #export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH
 #set this path at terminal
 #export XLA_FLAGS=--xla_gpu_cuda_data_dir=$CONDA_PREFIX 
 #set this for using adam

with tf.device('/GPU:0'):
    model.compile(
        optimizer='adam',
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits= True),
        metrics=['accuracy']
    )


'''
    model.fit(X_train_scaled, y_train, epochs = 30)
    print(model.evaluate(X_test_scaled, y_test))

'''


data_augmentation = keras.Sequential([
    layers.experimental.preprocessing.RandomFlip("horizontal", input_shape = (img_height, img_weight, 3)),
    layers.experimental.preprocessing.RandomRotation(.1),
    layers.experimental.preprocessing.RandomZoom(.1)  
])

num_classes = 5

model = Sequential([
  data_augmentation,
  layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Dropout(0.2),
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dense(num_classes)
])


with tf.device('/GPU:0'):
    model.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])
                
    model.fit(X_train_scaled, y_train, epochs=30)  

print(model.evaluate(X_test_scaled,y_test))





