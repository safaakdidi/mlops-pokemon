import sys
sys.path.append('PokemonProject')
from keras.preprocessing.image import ImageDataGenerator
import argparse
# from utils import save_data
from keras.utils import to_categorical
import os
import numpy as np
from PIL import Image

IMG_WIDTH = 64
IMG_HEIGHT = 64
BATCH_SIZE = 32

def run():
    # Create an ArgumentParser object
    parser = argparse.ArgumentParser(description='Preprocessing Pokemon Data')

    # Add arguments to the parser
    parser.add_argument('--data_dir', type=str, help='Enter the directory of the data')
    parser.add_argument('--output_dir_train', type=str, help='Enter the directory where to store augmented train data')
    parser.add_argument('--output_dir_val', type=str, help='Enter the directory where to store augmented val data')
    # Parse the arguments
    args = parser.parse_args()
    
    train_generator,val_generator=data_augmentation(args.data_dir)
    save_data(train_generator,args.output_dir_train,BATCH_SIZE,'train')
    save_data(val_generator,args.output_dir_val,BATCH_SIZE,'validation')








def data_augmentation(data_dir):
    train_datagen = ImageDataGenerator(
    rescale=1./255,
    #rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2,
    )


    train_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=(IMG_WIDTH, IMG_HEIGHT),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
    )


    val_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=(IMG_WIDTH, IMG_HEIGHT),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
    )
    return train_generator,val_generator

def save_data(dataset,output_dir,BATCH_SIZE,type):
  train_output_dir = os.path.join(output_dir, type)
  if not os.path.exists(train_output_dir):
    os.makedirs(train_output_dir)
  for i, (x, y) in enumerate(dataset):
    if i >= len(dataset):
        break
    for j in range(len(x)):
        img = x[j]
        label = y[j]
        filename = f'{i * BATCH_SIZE + j}.png'
        label_str = str(np.argmax(label))
        label_dir = os.path.join(train_output_dir, label_str)
        if not os.path.exists(label_dir):
            os.makedirs(label_dir)
        img_path = os.path.join(label_dir, filename)
        img = Image.fromarray((img * 255).astype('uint8'))
        if img.mode == 'P' and 'transparency' in img.info:
             img = img.convert('RGBA')
        img.save(img_path)



# train_generator,val_generator=data_augmentation('PokemonData')
# save_data(train_generator,'Output',BATCH_SIZE,'train')
# save_data(val_generator,'Output',BATCH_SIZE,'val')


if __name__ == '__main__':
    run()


    