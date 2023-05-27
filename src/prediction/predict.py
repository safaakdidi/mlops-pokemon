from keras.utils import load_img, img_to_array
import numpy as np
from constants import IMG_WIDTH, IMG_HEIGHT 

def predict_image(model, image_path):
  img = load_img(image_path, target_size=(IMG_WIDTH, IMG_HEIGHT))
  x = img_to_array(img)
  x = np.expand_dims(x, axis=0)

  images = np.vstack([x])
  preds = model.predict(images, batch_size=10)
  return preds