"""导入所需要的包"""
from keras.applications import ResNet50                         # 导入ResNet50
from keras.applications import InceptionV3                      # 导入InceptionV3
from keras.applications import Xception                         # 导入Xception
from keras.applications import VGG16                            # 导入VGG16
from keras.applications import VGG19                            # 导入VGG19
from keras.applications import imagenet_utils                   # 本模块的一些函数可以很方便的进行图像预处理和解码输出分类
from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
import numpy as np                                              # Numpy进行数值处理
import argparse
import cv2                                                      # 进行图像编辑
from sympy.abc import P
import h5py

#file=h5py.File('/Users/mac/Documents/GitHub/models/Imagenet_Download/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5','r')

from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input, decode_predictions
import numpy as np

model = VGG16(weights='imagenet')

img_path = '/Users/mac/Desktop/interim_report/test_img/elephant.jpeg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

preds = model.predict(x)
print('Predicted:', decode_predictions(preds, top=3)[0])
# Predicted: [(u'n02504013', u'Indian_elephant', 0.82658225), (u'n01871265', u'tusker', 0.1122357), (u'n02504458', u'African_elephant', 0.061040461)]
