
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import glob,cv2
from keras.utils.np_utils import *
import matplotlib.pyplot as plt

image_filenames = glob.glob("/Users/mac/Documents/GitHub/DataSet/3_class/*/*.jpg")
image_filenames = np.random.permutation(image_filenames)
#lables = list(map(lambda x:float(x.split('/')[-1].split('.')[0][:-4] == "airplane"),image_filenames))
def hot_code(lable):
    if lable == "airplane":
        lable = [1.0,0,0]
    elif lable == "bridge":
        lable = [0,1.0,0]
    elif lable == "lake":
        lable = [0,0,1.0]
    #lable = keras.utils.to_categorical(lable,3)
    return lable

lables = list(map(lambda x:x.split('/')[-1].split('.')[0][:-4],image_filenames))
hot_lables = list(map(hot_code,lables))

validation_size = int(0.2 * len(lables))

train_image_filenames = image_filenames[validation_size:]
train_hot_lables = hot_lables[validation_size:]

validation_image_filenames = image_filenames[:validation_size]
validation_hot_lables = hot_lables[:validation_size]

dataset = tf.data.Dataset.from_tensor_slices((train_image_filenames,train_hot_lables))
dataset_test = tf.data.Dataset.from_tensor_slices((validation_image_filenames,validation_hot_lables))

def _pre_read(img_filename,lable):
    image = tf.read_file(img_filename)
    image = tf.image.decode_jpeg(image,channels=3)
    image = tf.image.rgb_to_grayscale(image)
    image = tf.image.resize_images(image,(64,64),1)
    image = tf.reshape(image,[64,64,1])
    image = tf.image.per_image_standardization(image)
    lable = tf.reshape(lable,[3])
    return image,lable

dataset = dataset.map(_pre_read)
dataset = dataset.shuffle(300)
dataset = dataset.repeat()
dataset = dataset.batch(28)

dataset_test = dataset_test.map(_pre_read)
dataset_test = dataset_test.shuffle(300)
dataset_test = dataset_test.repeat()
dataset_test = dataset_test.batch(28)

model = keras.Sequential()
model.add(layers.Conv2D(64,(3,3),activation="relu",input_shape=(64,64,1)))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64,(3,3),activation="relu"))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64,(3,3),activation="relu"))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Flatten())
model.add(layers.Dense(64,activation="relu"))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(3,activation="softmax"))
model.summary()
model.compile(optimizer="adam",
              loss='binary_crossentropy',
              metrics=['acc'])

history = model.fit(dataset,epochs=30,steps_per_epoch=60,validation_data=dataset_test,validation_steps=15)
model.save_weights('/Users/mac/Documents/GitHub/models/2_26/class3_weights.h5')
plt.plot(history.epoch,history.history["acc"],'b',label="acc")

plt.plot(history.epoch,history.history["val_acc"],'r',label="Val_acc")
plt.xlabel("epoch")
plt.ylabel("acc")
plt.title("February_45_class45")
plt.savefig("/Users/mac/Documents/GitHub/models/2_26/acc_3_dropout.png")
plt.show()
plt.plot(history.epoch,history.history["val_loss"],'r',label="Val_loss")
plt.plot(history.epoch,history.history["loss"],'b',label="loss")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.title("February_45_class45")
plt.savefig("/Users/mac/Documents/GitHub/models/2_26/loss_3_dropout.png")
plt.show()