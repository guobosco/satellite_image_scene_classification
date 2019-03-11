import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import glob,cv2

image_filenames = glob.glob("/Users/mac/Documents/GitHub/DataSet/2_class/*/*.jpg")
image_filenames = np.random.permutation(image_filenames)
lables = list(map(lambda x:float(x.split('/')[-1].split('.')[0][:-4] == "airplane"),image_filenames))

dataset = tf.data.Dataset.from_tensor_slices((image_filenames,lables))

def _pre_read(img_filename,lable):
    image = tf.read_file(img_filename)
    image = tf.image.decode_jpeg(image,channels=3)
    image = tf.image.rgb_to_grayscale(image)
    image = tf.image.resize_images(image,(200,200),1)
    image = tf.reshape(image,[200,200,1])
    image = tf.image.per_image_standardization(image)
    lable = tf.reshape(lable,[1])
    return image,lable

dataset = dataset.map(_pre_read)
dataset = dataset.shuffle(300)
dataset = dataset.repeat()
dataset = dataset.batch(28)

model = keras.Sequential()
model.add(layers.Conv2D(64,(3,3),activation="relu",input_shape=(200,200,1)))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64,(3,3),activation="relu"))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64,(3,3),activation="relu"))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Flatten())
model.add(layers.Dense(64,activation="relu"))
model.add(layers.Dense(1,activation="sigmoid"))
model.summary()
model.compile(optimizer="adam",
              loss='binary_crossentropy',
              metrics=['acc'])
print(dataset)
model.fit(dataset,epochs=10,steps_per_epoch=50)
#model.train_on_batch()