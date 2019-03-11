import numpy as np
import os,random,shutil,glob
np.random.seed(7)

# 为Keras模型准备数据集
#1，指定一些超参数：
train_data_dir='/Users/mac/Documents/GitHub/DataSet/3_class'
val_data_dir='/Users/mac/Documents/GitHub/DataSet/3_class_test' # keras中将测试集称为validation set
train_samples_num=1500 # train set中全部照片数
val_samples_num=600
IMG_W,IMG_H,IMG_CH=150,150,3 # 单张图片的大小
batch_size=32
epochs=20  # 用比较少的epochs数目做演示，节约训练时间
class_num=3 # 此处有5个类别


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
    class_mode='categorical') # 此处是二分类问题，故而mode是binary

# 3，同样的方式准备测试集
val_datagen = ImageDataGenerator(rescale=1. / 255) # 只需要和trainset同样的scale即可，不需增强
val_generator = val_datagen.flow_from_directory(
        val_data_dir,
        target_size=(IMG_W, IMG_H),
        batch_size=batch_size,
        class_mode='categorical')

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
    model.add(Dense(class_num))  # 此处多分类问题，用Dense(class_num)
    model.add(Activation('softmax'))  # 多分类问题用softmax作为activation function

    # 模型的配置
    model.compile(loss='categorical_crossentropy',  # 定义模型的loss func，optimizer，
                  optimizer=optimizers.RMSprop(),  # 使用默认的lr=0.001
                  metrics=['accuracy'])  # 主要优化accuracy

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
    plt.savefig("/Users/mac/Documents/GitHub/models/2_26/use_VGG16/class_3.png")
    plt.show()

plot_training(history_ft)
