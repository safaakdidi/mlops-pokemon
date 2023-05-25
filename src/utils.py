import matplotlib.pyplot as plt
import os
import numpy as np
from keras.preprocessing.image import array_to_img

def plotImages(img_arr,label):

  for idx,img in enumerate(img_arr):
    if idx <= 10 :
      plt.figure(figsize=(5,5))
      plt.imshow(img)
      plt.title(img.shape)
      plt.axis = False
      plt.show()

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
        img = array_to_img(img)
        img.save(img_path)