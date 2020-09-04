import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np

fashion_mnist = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

train_images = train_images / 255.0

test_images = test_images / 255.0

train_images = train_images.reshape(train_images.shape[0], 28, 28, 1)

test_images = test_images.reshape(test_images.shape[0], 28, 28, 1)



model = keras.Sequential([
    keras.layers.Conv2D(64, kernel_size=(3,3), activation='relu', input_shape=(28,28,1)),
    keras.layers.Conv2D(32, kernel_size=(3,3), activation='relu'),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10)
])
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
model.fit(train_images, train_labels, epochs=10)
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

print('\nTest accuracy:', test_acc)
