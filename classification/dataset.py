import cv2
import glob
from sklearn.utils import shuffle
import numpy as np

def read_train_sets(train_path,classes,validation_size):
    """
    :param train_path:训练路径
    :param image_size: 图片大小
    :param classes: 类别
    :param validation_size:样本率
    :return: date_sets对象
    """
    class Datasets():
        pass
    #调用load_train函数
    images,labels,cls=load_train(train_path,classes)
    print("洗牌程序开始——————————>>>")
    images,labels,cls=shuffle(images,labels,cls)
    print("洗牌程序结束——————————>>>")

    if isinstance(validation_size,float):
        validation_size=int(validation_size*images.shape[0])

    validation_images = images[:validation_size]
    validation_labels = labels[:validation_size]
    validation_cls = cls[:validation_size]

    train_images = images[validation_size:]
    train_labels =  labels[validation_size:]
    train_cls = cls[validation_size:]


    class DataSets(object):
        pass
    data_sets = DataSets()

    data_sets.train = DataSet(train_images,train_labels,train_cls)
    data_sets.valid = DataSet(validation_images,validation_labels,validation_cls)

    return data_sets


def load_train(train_path,classes):
    """
    :param train_path:训练路径
    :param image_size: 图片大小
    :param classes: 类别
    :return: np数组格式的，images,labels,image_names,cls
    """
    images = []
    labels = []
    cls = []

    for clas in classes:
        index = classes.index(clas)
        image_filenames = glob.glob(train_path+"/"+clas+'/*.tif', recursive=True)
        for filename in image_filenames:
            label = np.zeros(len(classes))
            label[index] = 1.0
            labels.append(label)
            image = cv2.imread(filename)
            images.append(image)
            cls.append(clas)
    images = np.array(images)
    labels = np.array(labels)
    cls = np.array(cls)

    return images,labels,cls



class DataSet(object):

    def __init__(self,images,labels,cls):
        self._num_examples = images.shape[0]#图片的张数
        self._images = images
        self._labels = labels
        self._cls = cls
        self._epochs_done = 0#现在已经完成了几个
        self._index_in_epoch = 0#

    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels

    @property
    def cls(self):
        return self._cls

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_done(self):
        return self._epochs_done



    def next_batch(self,batch_size):

        start = self._index_in_epoch
        self._index_in_epoch+=batch_size

        if self._index_in_epoch>self._num_examples:

            self._epochs_done+=1
            start = 0
            self._index_in_epoch=batch_size
            assert batch_size<=self._num_examples
        end=self._index_in_epoch

        return self._images[start:end],self._labels[start:end],self._cls[start:end]






















