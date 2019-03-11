import numpy as np
import os,random,shutil,glob
np.random.seed(7)

# 为Keras模型准备数据集
#1，指定一些超参数：
train_data_dir='/Users/mac/Documents/GitHub/DataSet/2_class'
val_data_dir='/Users/mac/Documents/GitHub/DataSet/2_class_test' # keras中将测试集称为validation set
train_samples_num=1000 # train set中全部照片数
val_samples_num=400
IMG_W,IMG_H,IMG_CH=150,150,3 # 单张图片的大小
batch_size=32
epochs=2  # 用比较少的epochs数目做演示，节约训练时间

# 2，准备训练集，keras有很多Generator可以直接处理图片的加载，增强等操作，封装的非常好
from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator( # 单张图片的处理方式，train时一般都会进行图片增强
        rescale=1. / 255, # 图片像素值为0-255，此处都乘以1/255，调整到0-1之间
        shear_range=0.2, # 斜切
        zoom_range=0.2, # 放大缩小范围
        horizontal_flip=True) # 水平翻转

train_generator = train_datagen.flow_from_directory(# 从文件夹中产生数据流
    train_data_dir, # 训练集图片的文件夹
    target_size=(IMG_W, IMG_H), # 调整后每张图片的大小
    batch_size=batch_size,
    class_mode='binary') # 此处是二分类问题，故而mode是binary

# 3，同样的方式准备测试集
val_datagen = ImageDataGenerator(rescale=1. / 255) # 只需要和trainset同样的scale即可，不需增强
val_generator = val_datagen.flow_from_directory(
        val_data_dir,
        target_size=(IMG_W, IMG_H),
        batch_size=batch_size,
        class_mode='binary')

# 4，建立Keras模型：模型的建立主要包括模型的搭建，模型的配置
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import optimizers


def build_model(input_shape):
    # 模型的搭建：此处构建三个CNN层+2个全连接层的结构
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))  # Dropout防止过拟合
    model.add(Dense(1))  # 此处虽然是二分类，但是不能用Dense(2)，因为后面的activation是sigmoid，这个函数只能输出一个值，即class_0的概率
    model.add(Activation('sigmoid'))  # 二分类问题用sigmoid作为activation function

    # 模型的配置
    model.compile(loss='binary_crossentropy',  # 定义模型的loss func，optimizer，
                  optimizer=optimizers.RMSprop(lr=0.0001),
                  metrics=['accuracy'])  # 主要优化accuracy
    # 二分类问题的loss function使用binary_crossentropy，此处使用准确率作为优化目标
    return model  # 返回构建好的模型

model=build_model(input_shape=(IMG_W,IMG_H,IMG_CH)) # 输入的图片维度
# 模型的训练
history_ft = model.fit_generator(train_generator, # 数据流
                        steps_per_epoch=train_samples_num // batch_size,
                        epochs=epochs,
                        validation_data=val_generator,
                        validation_steps=val_samples_num // batch_size)

# 画图，将训练时的acc和loss都绘制到图上
import matplotlib.pyplot as plt


def plot_training(history):
    plt.figure(12)

    plt.subplot(121)
    train_acc = history.history['acc']
    val_acc = history.history['val_acc']
    epochs = range(len(train_acc))
    plt.plot(epochs, train_acc, 'b', label='train_acc')
    plt.plot(epochs, val_acc, 'r', label='test_acc')
    plt.title('Train and Test accuracy')
    plt.legend()

    plt.subplot(122)
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(len(train_loss))
    plt.plot(epochs, train_loss, 'b', label='train_loss')
    plt.plot(epochs, val_loss, 'r', label='test_loss')
    plt.title('Train and Test loss')
    plt.legend()

    plt.show()

plot_training(history_ft)


# 用训练好的模型来预测新样本
from PIL import Image
from keras.preprocessing import image
def predict(model, img_path, target_size):
    img=Image.open(img_path) # 加载图片
    if img.size != target_size:
        img = img.resize(target_size)

    x = image.img_to_array(img)
    x *=1./255 # 相当于ImageDataGenerator(rescale=1. / 255)
    x = np.expand_dims(x, axis=0) # 调整图片维度
    preds = model.predict(x) # 预测
    return preds[0]

print(predict(model,'/Users/mac/Documents/GitHub/DataSet/2_class_test/airplane/airplane_020.jpg',(IMG_W,IMG_H)))
print(predict(model,'/Users/mac/Documents/GitHub/DataSet/2_class_test/lake/lake_018.jpg',(IMG_W,IMG_H)))

# 预测一个文件夹中的所有图片
#new_sample_gen=ImageDataGenerator(rescale=1. / 255)
#newsample_generator=new_sample_gen.flow_from_directory(
#        '/Users/mac/Documents/GitHub/DataSet/2_class_test',
#        target_size=(IMG_W, IMG_H),
#        batch_size=16,
#        class_mode=None,
#        shuffle=False)
#predicted=model.predict_generator(newsample_generator)
#print(predicted)

# 模型保存
# model.save_weights('/Users/mac/Documents/GitHub/models/2_26/Model2_weights.h5') # 这个只保存weights，不保存模型的结构
model.save('/Users/mac/Documents/GitHub/models/2_26/Model2.h5') # 对于一个完整的模型，应该要保存这个

from keras.models import load_model
saved_model=load_model('/Users/mac/Documents/GitHub/models/2_26/Model2.h5')
print(predict(saved_model,'/Users/mac/Documents/GitHub/DataSet/2_class_test/airplane/airplane_020.jpg',(IMG_W,IMG_H)))
print(predict(saved_model,'/Users/mac/Documents/GitHub/DataSet/2_class_test/lake/lake_018.jpg',(IMG_W,IMG_H)))