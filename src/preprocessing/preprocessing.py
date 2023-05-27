import tensorflow as tf


def preprocess_split_data(data_dir, img_height, img_width, batch_size, validation_split):

  train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    seed=123,
    label_mode='categorical',
    shuffle=True,
    validation_split=validation_split,
    subset="training",
    image_size=(img_height, img_width),
    batch_size=batch_size)

  val_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    seed=123,
    label_mode='categorical',
    shuffle=True,
    validation_split=validation_split,
    subset="validation",
    image_size=(img_height, img_width),
    batch_size=batch_size)

  return train_ds, val_ds