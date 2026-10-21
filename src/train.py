"""
AI-Generated vs Real Image Classification (CNN, TensorFlow/Keras)

Originally written as a Google Colab notebook. Run it in Colab, or locally after
downloading the dataset yourself (see README.md -> "Getting the dataset").

Colab setup commands used in the original notebook (run these in a cell, not in this file):
    !mkdir ~/.kaggle/
    !cp "/content/kaggle(1).json" ~/.kaggle/kaggle.json
    !chmod 600 ~/.kaggle/kaggle.json
    !kaggle datasets download -d cashbowman/ai-generated-images-vs-real-images
    !unzip -qq "ai-generated-images-vs-real-images"
    !pip install Pillow
"""

import tensorflow as tf
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import glob
import PIL
from PIL import Image

# to resize images to 64x64
source="/content/data/RealArt/RealArt";
destination="/content/data/RealArt/RealArt"

# resizing function

i=1;
for img in glob.glob(source+'/*.*'):
  try:
    puffin_image = Image.open(img);
    resized_img=puffin_image.resize((64,64))
    resized_img = np.array(resized_img)
    cv2.imwrite(img, resized_img)
    i=i+1;
  except:
    print('{} is not saved'.format(img))

puffin_image = Image.open('/content/data/RealArt/RealArt/-man-sits-with-a-woman-on-her-phone-at-a-table-while-looking-at-a-computer_l.jpg')
print(f"Image size (width, height): {puffin_image.size}")
puffin_image

data = tf.keras.utils.image_dataset_from_directory('data', image_size=(64, 64))

data_it = data.as_numpy_iterator()

batch = data_it.next()

fig, ax = plt.subplots(ncols=8, figsize=(10, 10))
for idx, img in enumerate(batch[0][:8]):
    ax[idx].imshow(img.astype(int))
    ax[idx].title.set_text(batch[1][idx])


print(batch[0].min())
print(batch[0].max())

data = data.map(lambda x, y : (x/255, y))

scaled_it = data.as_numpy_iterator()

batch = scaled_it.next()

print(batch[0].min())
print(batch[0].max())

len(data)

train_size = int(len(data)*0.7)
cv_size = int(len(data)*0.2)
test_size = int(len(data)*0.1)+1

train_size+cv_size+test_size

train = data.take(train_size)
cv = data.skip(train_size).take(cv_size)
test = data.skip(train_size+cv_size).take(test_size)

len(train)+len(cv)+len(test)




from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten

model = Sequential()

model.add(Conv2D(16, (4, 4), 1, activation='relu', input_shape=(64, 64, 3)))
model.add(MaxPooling2D())

model.add(Conv2D(32, (4, 4), 1, activation='relu'))
model.add(MaxPooling2D())

model.add(Conv2D(16, (4, 4), 1, activation='relu'))
model.add(MaxPooling2D())

model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile('adam', loss = tf.losses.BinaryCrossentropy(), metrics=['accuracy'])

model.summary()

logdir = 'logs'

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)

hist = model.fit(train, epochs=20, validation_data=cv, callbacks=[tensorboard_callback])

#initial accuracy

grf=plt.figure();
plt.plot(hist.history['accuracy'], color='blue',label='training accuracy')
plt.plot(hist.history['val_accuracy'],color='yellow',label='testing accuracy')
grf.suptitle('Accuracy', fontsize=20)
plt.legend(loc='lower right')
plt.show()

# initial accuarcy

grf=plt.figure();
plt.plot(hist.history['loss'], color='blue',label='training loss')
plt.plot(hist.history['val_loss'],color='red',label='testing loss')
grf.suptitle('Loss', fontsize=20)
plt.legend(loc='upper right')
plt.show()

model.save('cnn_model.h5')
