import tensorflow as tf
import glob
import random
import numpy as np
import cv2
from sklearn.utils import shuffle

train_path = "/Users/mac/Documents/GitHub/DataSet/UC_Merge_LandUse/images_uni_224"
classes =["agricultural","airplane","baseballdiamond",
          "beach","buildings","chaparral","denseresidential",
          "forest","freeway","golfcourse","harbor","intersection",
          "mediumresidential","mobilehomepark","overpass","parkinglot",
          "river","runway","sparseresidential","storagetanks","tenniscourt"]
images = []
labels = []
img_names = []
cls = []
for clas in classes:
    index = classes.index(clas)
    image_filenames = glob.glob(train_path+"/"+clas+'/*.tif', recursive=True)
    img_names.append(image_filenames)
    for filename in image_filenames:
        label = np.zeros(len(classes))
        label[index] = 1.0
        labels.append(label)
        image = cv2.imread(filename)
#        image = image.astype(np.float32)
        image = np.multiply(image,1.0/255.0)
        images.append(image)
        cls.append(clas)
images = np.array(images)
labels = np.array(labels)
img_names = np.array(img_names)
cls = np.array(cls)

#对数据执行洗牌程序
images,labels,img_names,cls=shuffle(images,labels,img_names,cls)
images,labels,cls=shuffle(images,labels,cls)

#截取训练数据集和测试数据集
validation_rate = 0.2
if isinstance(validation_rate, float):
    validation_size = int(validation_rate * images.shape[0])
validation_images = images[:validation_size]
validation_labels = labels[:validation_size]
validation_cls = cls[:validation_size]

train_images = images[validation_size:]
train_labels =  labels[validation_size:]
train_cls = cls[validation_size:]


dataset_train = tf.data.Dataset.from_tensor_slices((train_images,train_labels))
dataset_validation = tf.data.Dataset.from_tensor_slices((validation_images,validation_labels))
print("训练集的数量：\t{}".format(len(dataset_train.train.train_labels)))
print("测试集的数量：\t{}".format(len(dataset_validation.validation_labels)))

