import numpy as np
import os,random,shutil
np.random.seed(7)

# 为Keras模型准备数据集
#1，指定一些超参数：
FOLDER='/Users/mac/Documents/GitHub/DataSet'
train_data_dir=os.path.join(FOLDER,'NWPU')
val_data_dir=os.path.join(FOLDER,'NWPU_test')
train_samples_num=22500 # train set中全部照片数
val_samples_num=9000
IMG_W,IMG_H,IMG_CH=150,150,3 # 单张图片的大小
batch_size=50 # 这儿要保证400和100能除断
epochs=50  # 用比较少的epochs数目做演示，节约训练时间
class_num=45 # 此处有5个类别

save_folder='/Users/mac/Documents/GitHub/models/summary' # bottleneck特征保存位置

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
        class_mode='categorical', # 这个地方要修改，要不然出错
        shuffle=False) # 必须为False，否则顺序打乱之后，和后面的label对应不上。
    bottleneck_features_train = model.predict_generator(
        generator, train_samples_num // batch_size)
    np.save(os.path.join(save_folder,'vgg16_bottleneck_features_train.npy'), bottleneck_features_train)
    print('bottleneck features of train set is saved.')

    generator = datagen.flow_from_directory(
        val_data_dir,
        target_size=(IMG_W, IMG_H),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False)
    bottleneck_features_validation = model.predict_generator(
        generator, val_samples_num // batch_size)
    np.save(os.path.join(save_folder,'vgg16_bottleneck_features_val.npy'),bottleneck_features_validation)
    print('bottleneck features of test set is saved.')

print('start save_bottlebeck_features().....')
save_bottlebeck_features()
print('End save_bottlebeck_features().....')

def my_model():
    '''
    自定义一个模型，该模型仅仅相当于一个分类器，只包含有全连接层，对提取的特征进行分类即可
    :return:
    '''
    # 模型的结构
    model = Sequential()
    model.add(Flatten(input_shape=train_data.shape[1:])) # 将所有data进行flatten
    model.add(Dense(512, activation='relu')) # 256个全连接单元
    model.add(Dropout(0.5)) # dropout正则
    model.add(Dense(class_num, activation='softmax')) # 与二分类不同之处：要用Dense(class_num)和softmax

    # 模型的配置
    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy']) # model的optimizer等

    return model

from keras.utils import to_categorical
# 只需要训练分类器模型即可，不需要训练特征提取器
train_data = np.load(os.path.join(save_folder,'vgg16_bottleneck_features_train.npy')) # 加载训练图片集的所有图片的VGG16-notop特征
train_labels = np.array([0] * 500 + [1] * 500+ [2] * 500 + [3] * 500 + [4] * 500
                        + [5] * 500 + [6] * 500 + [7] * 500+ [8] * 500 + [9] * 500 + [10] * 500
                        + [11] * 500 + [12] * 500 + [13] * 500+ [14] * 500 + [15] * 500
                        + [16] * 500+ [17] * 500 + [18] * 500 + [19] * 500+ [20] * 500
                        + [21] * 500 + [22] * 500+ [23] * 500 + [24] * 500 + [25] * 500
                        + [26] * 500 + [27] * 500 + [28] * 500 + [29] * 500 + [30] * 500
                        + [31] * 500 + [32] * 500 +[33] * 500 + [34] * 500 + [35] * 500
                        + [36] * 500 + [37] * 500 +[38] * 500 + [39] * 500 + [40] * 500
                        + [41] * 500 + [42] * 500 +[43] * 500 + [44] * 500 )
# label是每个类别500张图片，共45个类别
# 设置标签，并规范成Keras默认格式
train_labels = to_categorical(train_labels, class_num)

validation_data = np.load(os.path.join(save_folder,'vgg16_bottleneck_features_val.npy'))
validation_labels = np.array([0] * 200 + [1] * 200+ [2] * 200 + [3] * 200 + [4] * 200
                        + [5] * 200 + [6] * 200 + [7] * 200+ [8] * 200 + [9] * 200 + [10] * 200
                        + [11] * 200 + [12] * 200 + [13] * 200+ [14] * 200 + [15] * 200
                        + [16] * 200+ [17] * 200 + [18] * 200 + [19] * 200+ [20] * 200
                        + [21] * 200 + [22] * 200+ [23] * 200 + [24] * 200 + [25] * 200
                        + [26] * 200 + [27] * 200 + [28] * 200 + [29] * 200 + [30] * 200
                        + [31] * 200 + [32] * 200 +[33] * 200 + [34] * 200 + [35] * 200
                        + [36] * 200 + [37] * 200 +[38] * 200 + [39] * 200 + [40] * 200
                        + [41] * 200 + [42] * 200 +[43] * 200 + [44] * 200)
validation_labels = to_categorical(validation_labels, class_num)

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
    plt.savefig("/Users/mac/Documents/GitHub/models/summary/vgg16_NWPU45.png")
    plt.show()

plot_training(history_ft)
clf_model.save('/Users/mac/Documents/GitHub/models/summary/vgg16_NWPU45.h5')