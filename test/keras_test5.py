import numpy as np
import os,random,shutil
np.random.seed(7)

# 为Keras模型准备数据集
#1，指定一些超参数：
FOLDER='/Users/mac/Documents/GitHub/DataSet'
train_data_dir=os.path.join(FOLDER,'3_class')
val_data_dir=os.path.join(FOLDER,'3_class_test')
train_samples_num=1500 # train set中全部照片数
val_samples_num=600
IMG_W,IMG_H,IMG_CH=150,150,3 # 单张图片的大小
batch_size=50 # 这儿要保证400和100能除断
epochs=50  # 用比较少的epochs数目做演示，节约训练时间
class_num=3 # 此处有5个类别

save_folder='/Users/mac/Documents/GitHub/models/2_26/use_VGG16_class3' # bottleneck特征保存位置

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
    np.save(os.path.join(save_folder,'bottleneck_features_train.npy'), bottleneck_features_train)
    print('bottleneck features of train set is saved.')

    generator = datagen.flow_from_directory(
        val_data_dir,
        target_size=(IMG_W, IMG_H),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False)
    bottleneck_features_validation = model.predict_generator(
        generator, val_samples_num // batch_size)
    np.save(os.path.join(save_folder,'bottleneck_features_val.npy'),bottleneck_features_validation)
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
    model.add(Dense(256, activation='relu')) # 256个全连接单元
    model.add(Dropout(0.5)) # dropout正则
    model.add(Dense(class_num, activation='softmax')) # 与二分类不同之处：要用Dense(class_num)和softmax

    # 模型的配置
    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy']) # model的optimizer等

    return model

from keras.utils import to_categorical
# 只需要训练分类器模型即可，不需要训练特征提取器
train_data = np.load(os.path.join(save_folder,'bottleneck_features_train.npy')) # 加载训练图片集的所有图片的VGG16-notop特征
train_labels = np.array([0] * 500 + [1] * 500+ [2] * 500)
# label是每个类别500张图片，共3个类别
# 设置标签，并规范成Keras默认格式
train_labels = to_categorical(train_labels, class_num)

validation_data = np.load(os.path.join(save_folder,'bottleneck_features_val.npy'))
validation_labels = np.array([0] * 200 + [1] * 200+ [2]*200)
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
    plt.savefig("/Users/mac/Documents/GitHub/models/2_26/use_VGG16_class3/class_3.png")
    plt.show()

plot_training(history_ft)
clf_model.save('/Users/mac/Documents/GitHub/models/2_26/use_VGG16_class3/class_3_model')