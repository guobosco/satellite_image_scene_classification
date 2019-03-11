import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import glob,cv2
from keras.utils.np_utils import *
import matplotlib.pyplot as plt

image_filenames = glob.glob("/Users/mac/Documents/GitHub/DataSet/NWPU/*/*.jpg")
image_filenames = np.random.permutation(image_filenames)
lables = list(map(lambda x:x.split('/')[-1].split('.')[0][:-4],image_filenames))
unique_lable = list(set(lables))
hot_lables = np.zeros((len(lables),len(unique_lable)))
for i,lable in enumerate(lables):
    hot_lables[i,unique_lable.index(lable)] =1

print(hot_lables)
dataset = tf.data.Dataset.from_tensor_slices((image_filenames,hot_lables))

def _pre_read(img_filename,lable):
    image = tf.read_file(img_filename)
    image = tf.image.decode_jpeg(image,channels=3)
    image = tf.image.rgb_to_grayscale(image)
    image = tf.image.resize_images(image,(200,200),1)
    image = tf.reshape(image,[200,200,1])
    image = tf.image.per_image_standardization(image)
    lable = tf.reshape(lable,[45])
    return image,lable

dataset = dataset.map(_pre_read)
dataset = dataset.shuffle(300)
dataset = dataset.repeat()
dataset = dataset.batch(32)

model = keras.Sequential()
model.add(layers.Conv2D(64,(3,3),activation="relu",input_shape=(200,200,1)))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64,(3,3),activation="relu"))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64,(3,3),activation="relu"))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Flatten())
model.add(layers.Dense(128,activation="relu"))
model.add(layers.Dense(45,activation="softmax"))
model.summary()
model.compile(optimizer="adam",
              loss='binary_crossentropy',
              metrics=['acc'])
print(dataset)
history = model.fit(dataset,epochs=20,steps_per_epoch=984,validation_data = dataset_test)
#model.train_on_batch()
plt.plot(history.epoch,history.history["loss"],'r')
plt.plot(history.epoch,history.history["acc"],'b')
plt.title("February_25_class45")
plt.savefig("/Users/mac/Documents/GitHub/models/2_25/acc_loss_45.png")
plt.show()