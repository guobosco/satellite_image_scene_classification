import tensorflow as tf
import glob
import numpy as np

"""
读入数据模块：
1. 创建Dataset
2. 处理Dataset
3. 构建iterator(迭代器)
"""
image_filenames = glob.glob('./classification/Images/*/*.tif',recursive=True)
image_filenames = np.random.permutation(image_filenames)

classes =["agricultural","airplane","baseballdiamond",
          "beach","buildings","chaparral","denseresidential",
          "forest","freeway","golfcourse","harbor","intersection",
          "mediumresidential","mobilehomepark","overpass","parkinglot",
          "river","runway","sparseresidential","storagetanks","tenniscourt"]
for index in range(0,len(classes)):
    labels = [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]

for image_filename in image_filenames:
    for index in range(0,len(classes)):
        if classes[index] in image_filename:
            print(classes[index])
            labels[index] = 1.0
            labeled_image_filenames = list(map(labels,image_filenames))
            print(labels)
            print(labeled_image_filenames)


def load_train(train_path,image_size,classes):
    images = []
    labels = []
    img_names = []
    cls = []

    print("开始读训练数据")
    for fields in classes:
        index = classes.index(fields)
        print('现在开始读{}空间:index值是{}'.format(fields,index))
        path = os.path.join(train_path,fields)
        img_list = os.listdir(path)
        img_list_name = []
        for i in range(0, len(img_list)):
            img_list_name.append(os.path.basename(img_list[i]))
        for f1 in img_list_name:
            image = cv2.imread(path+"/"+f1)
            image = cv2.resize(image,(image_size,image_size),0,0,cv2.INTER_LINEAR)
            #image = image.astype(np.float32)
            image = np.multiply(image,1.0/255.0)#归一化
            images.append(image)
            label = np.zeros(len(classes))
            label[index]=1.0
            labels.append(label)
            flbase = os.path.basename(f1)
            img_names.append(flbase)
            cls.append(fields)
    images = np.array(images)
    labels = np.array(labels)
    img_names = np.array(img_names)
    cls = np.array(cls)
    return images,labels,img_names,cls
