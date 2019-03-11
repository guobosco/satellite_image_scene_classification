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
batch_size=8
epochs=10  # 用比较少的epochs数目做演示，节约训练时间

"""首先使用预训练好的模型VGG16来提取train set和test set图片的特征，
然后将这些特征保存，这些特征实际上就是numpy.ndarray，故而可以保存为数字，然后加载这些数字来训练。"""
# 此处的训练集和测试集并不是原始图片的train set和test set，而是用VGG16对图片提取的特征，这些特征组成新的train set和test set
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras import applications
def save_bottlebeck_features():
    datagen = ImageDataGenerator(rescale=1. / 255) # 不需图片增强

    # build the VGG16 network
    model = applications.VGG16(include_top=False, weights='imagenet')
    # 使用imagenet的weights作为VGG16的初始weights,由于只是特征提取，故而只取前面的卷积层而不需要DenseLayer，故而include_top=False

    generator = datagen.flow_from_directory( # 产生train set
        train_data_dir,
        target_size=(IMG_W, IMG_H),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False) # 必须为False，否则顺序打乱之后，和后面的label对应不上。
    bottleneck_features_train = model.predict_generator(
        generator, train_samples_num // batch_size) # 如果是32，这个除法得到的是62，抛弃了小数，故而得到1984个sample
    np.save('/Users/mac/Documents/GitHub/models/2_26/use_VGG16/bottleneck_features_train.npy', bottleneck_features_train)
    print('bottleneck features of train set is saved.')

    generator = datagen.flow_from_directory(
        val_data_dir,
        target_size=(IMG_W, IMG_H),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)
    bottleneck_features_validation = model.predict_generator(
        generator, val_samples_num // batch_size)
    np.save('/Users/mac/Documents/GitHub/models/2_26/use_VGG16/bottleneck_features_val.npy',bottleneck_features_validation)
    print('bottleneck features of test set is saved.')

save_bottlebeck_features()

def my_model():
    '''
    自定义一个模型，该模型仅仅相当于一个分类器，只包含有全连接层，对提取的特征进行分类即可
    :return:
    '''
    # 模型的结构
    model = Sequential()
    model.add(Flatten(input_shape=train_data.shape[1:])) # 将所有data进行flatten
    model.add(Dense(256, activation='relu')) # 256个全连接单元
    model.add(Dropout(0.5)) # dropout正则
    model.add(Dense(1, activation='sigmoid')) # 此处定义的模型只有后面的全连接层，由于是本项目特殊的，故而需要自定义

    # 模型的配置
    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy', metrics=['accuracy']) # model的optimizer等

    return model

# 只需要训练分类器模型即可，不需要训练特征提取器
train_data = np.load('/Users/mac/Documents/GitHub/models/2_26/use_VGG16/bottleneck_features_train.npy') # 加载训练图片集的所有图片的VGG16-notop特征
train_labels = np.array(
    [0] * int((train_samples_num / 2)) + [1] * int((train_samples_num / 2)))
# label是1000个cat，1000个dog，由于此处VGG16特征提取时是按照顺序，故而[0]表示cat，1表示dog

validation_data = np.load('/Users/mac/Documents/GitHub/models/2_26/use_VGG16/bottleneck_features_val.npy')
validation_labels = np.array(
    [0] * int((val_samples_num / 2)) + [1] * int((val_samples_num / 2)))

# 构建分类器模型
clf_model=my_model()
history_ft = clf_model.fit(train_data, train_labels,
              epochs=epochs,
              batch_size=batch_size,
              validation_data=(validation_data, validation_labels))

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
    plt.savefig("/Users/mac/Documents/GitHub/models/2_26/use_VGG16/class_2.png")
    plt.show()

plot_training(history_ft)
# 将本模型保存一下
clf_model.save('/Users/mac/Documents/GitHub/models/2_26/use_VGG16/top_FC_model')