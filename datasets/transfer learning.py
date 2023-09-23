import numpy as np
import cv2

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras 
from keras import layers
from keras.models import Sequential
import tensorflow_hub as hub
import pathlib

IMAGE_SHAPE = (224, 224)
feature_extractor_model = "https://tfhub.dev/google/imagenet/mobilenet_v2_100_160/feature_vector/5"


data_dir = "./datasets/flower_photos"

data_dir = pathlib.Path(data_dir)

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

for flower_name, images in flowers_images_dict.items():
    for image in images:
        img = cv2.imread(str(image))
        resized_img = cv2.resize(img,(160,160))
        X.append(resized_img)
        y.append(flowers_labels_dict[flower_name])

X = np.array(X)
y = np.array(y)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

X_train_scaled = X_train / 255.0
X_test_scaled = X_test / 255.0

#tensor hub has a lot of pre trained models
#also the package tensorhub gives a module just to import the pretrained model with the link
#in their website the input shape is specified to use those models
#if not the correct size as input, then many problems... so always verify the input size spec from the website for any pretrained model

pretrained_model_without_top_layer = hub.KerasLayer(feature_extractor_model, input_shape=(160, 160, 3), trainable=False)

num_of_flowers = 5
model = tf.keras.Sequential([
  pretrained_model_without_top_layer,
  tf.keras.layers.Dense(num_of_flowers)
])
with tf.device('/GPU:0'):
    model.compile(
    optimizer="adam",
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'])

    model.fit(X_train_scaled, y_train, epochs=5)

print(model.evaluate(X_test_scaled,y_test))