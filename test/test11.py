import numpy as np
import glob,cv2
from sklearn.utils import shuffle
import keras
from keras import layers
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Dropout, Dense, Flatten
from keras.models import Sequential

image_size = 64
validation_rate = 0.2
train_path = "/Users/mac/Documents/GitHub/DataSet/UC_Merge_LandUse/images_uni_224"
image_filenames = glob.glob(train_path + '/*/*.tif', recursive=True)

lables = list(map(lambda x:x.split('/')[-1].split('.')[0][:-2],image_filenames))
classes =list(set(lables))

images = []
labels = []
cls = []
print("正在读取数据————>")
for clas in classes:
    index = classes.index(clas)
    image_filenames = glob.glob(train_path+"/"+clas+'/*.tif', recursive=True)
    for filename in image_filenames:
        label = np.zeros(len(classes))
        label = label.astype(np.float32)
        label[index] = 1.0
        labels.append(label)
        image = cv2.imread(filename)
        image = cv2.resize(image,(image_size,image_size),0,0,cv2.INTER_LINEAR)
        image = image.astype(np.float32)
        image = np.multiply(image,1.0/255.0)
        images.append(image)
        cls.append(clas)
print("读取数据完成")
images = np.array(images)
labels = np.array(labels)
cls = np.array(cls)
images,labels,cls=shuffle(images,labels,cls)

if isinstance(validation_rate, float):
    validation_size = int(validation_rate * images.shape[0])

validation_images = images[:validation_size]
validation_labels = labels[:validation_size]
validation_cls = cls[:validation_size]

train_images = images[validation_size:]
train_labels = labels[validation_size:]
train_cls = cls[validation_size:]
train_images = train_images.reshape(-1,64*64)
"""
train_dataset = tf.data.Dataset.from_tensor_slices((train_images,train_labels))       #本方法可以从np.arrray，list,tensor等创建一个dataset
train_dataset = train_dataset.shuffle(300)
train_dataset = train_dataset.repeat(-1)
train_dataset = train_dataset.batch(64)

# 构建迭代器
iterator = train_dataset.make_one_shot_iterator()
image_batch,labels_batch = iterator.get_next()
"""
#初始化模型
model = keras.Sequential()

#添加层，构建网络
model = Sequential()
# 读入的图片尺寸是灰度图，只有一个通道
model.add(Convolution2D(32, (3, 3), input_shape=(64, 64, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Convolution2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Convolution2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
# this converts our 3D feature maps to 1D feature vectors
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(21, activation='softmax'))

model.summary()

#编译模型，添加优化函数

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

#训练模型
model.fit(train_images,train_labels,epochs=50,batch_size=52)

#评价模型
model.evaluate(validation_images,validation_labels)

#预测
model.predict(validation_images[:10])
