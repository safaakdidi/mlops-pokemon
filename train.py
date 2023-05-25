import keras
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.utils import to_categorical
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, GlobalAveragePooling2D
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.models import Sequential
from keras import applications
from keras.utils import plot_model
import tensorflow as tf
import cv2

IMG_WIDTH = 64
IMG_HEIGHT = 64
BATCH_SIZE = 32
base_dir = 'PokemonData'

# ImageDataGenerator class is used to perform image augmentation and normalization
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    # rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2
)
# categorical =>  dataset contains multiple classes
# flow_from_directory => load the images from the subdirectories
# subset => split the dataset into training and validation sets
# batch_size => number of images in each training/validation batch
# training
train_generator = train_datagen.flow_from_directory(
    base_dir,
    target_size=(IMG_WIDTH, IMG_HEIGHT),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

# testing
val_generator = train_datagen.flow_from_directory(
    base_dir,
    target_size=(IMG_WIDTH, IMG_HEIGHT),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)
print(train_generator.class_indices)


def plotImages(img_arr, label):
    for idx, img in enumerate(img_arr):
        if idx <= 10:
            plt.figure(figsize=(5, 5))
            plt.imshow(img)
            plt.title(img.shape)
            plt.axis = False
            plt.show()


t_img, label = train_generator.next()
plotImages(t_img, label)

# model building

train_data_encoded = to_categorical(train_generator.labels)
train_data_encoded

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=7)
filepath = "model.h5"
ckpt = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
rlp = ReduceLROnPlateau(monitor='loss', patience=3, verbose=1)

# defining model
image_size = (64, 64, 3)


def cnn(image_size, num_classes):
    classifier = Sequential()
    classifier.add(Conv2D(64, (5, 5), input_shape=image_size, activation='relu', padding='same'))
    classifier.add(MaxPooling2D(pool_size=(2, 2)))
    classifier.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    classifier.add(MaxPooling2D(pool_size=(2, 2)))
    classifier.add(Flatten())
    classifier.add(Dense(num_classes, activation='softmax'))
    classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
    return classifier


num_classes = len(train_generator.class_indices)
neuralnetwork_cnn = cnn(image_size, num_classes)
neuralnetwork_cnn.summary()
plot_model(neuralnetwork_cnn, show_shapes=True)

history = neuralnetwork_cnn.fit_generator(
    generator=train_generator, validation_data=val_generator,
    callbacks=[es, ckpt, rlp], epochs=50,
)

fig, ax = plt.subplots(figsize=(20, 6))
pd.DataFrame(history.history).iloc[:, :-1].plot(ax=ax)

test_loss, test_acc = neuralnetwork_cnn.evaluate(train_generator)
print('Test accuracy:', test_acc)

h = history.history

# Visualizing loss
plt.plot(h['loss'], 'r', label='Loss')
plt.plot(h['val_loss'], 'b', label='Val Loss')
plt.legend()
plt.show()

image_to_predict = cv2.imread(base_dir + '/Pikachu/00000001.png', cv2.IMREAD_COLOR)
plt.imshow(cv2.cvtColor(image_to_predict, cv2.COLOR_BGR2RGB))
plt.show()
img_to_predict = np.expand_dims(cv2.resize(image_to_predict, (64, 64)), axis=0)
res = neuralnetwork_cnn.predict(img_to_predict)
predicted_class_index = np.argmax(res)
probs = tf.nn.softmax(res, axis=-1)
print(probs)

model = tf.keras.models.load_model('model.h5',
                                   custom_objects={'CategoricalCrossentropy': tf.losses.CategoricalCrossentropy})

test_loss, test_acc = model.evaluate(val_generator)
print('Test accuracy:', test_acc)

image_to_predict = cv2.imread(base_dir + '/Pikachu/00000001.png', cv2.IMREAD_COLOR)
plt.imshow(cv2.cvtColor(image_to_predict, cv2.COLOR_BGR2RGB))
plt.show()
img_to_predict = np.expand_dims(cv2.resize(image_to_predict, (64, 64)), axis=0)
res = model.predict(img_to_predict)
predicted_class_index = np.argmax(res)
# probs = tf.nn.softmax(res, axis=-1)
print(predicted_class_index)
