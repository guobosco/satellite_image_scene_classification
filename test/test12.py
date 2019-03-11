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

"""解析命令行参数,只需要一个命令行参数--image,这是要分类的输入图像的路径，还可以接受一个可选的命令行参数--model，指定想要使用的预训练模型，默认使用VGG16"""
ap = argparse.ArgumentParser()
ap.add_argument("-i","--image",required=True,
                help = "要分类的输入图片的路径")
ap.add_argument("-model","--model",type=str,default="vgg16",
                help = "指定要使用的预训练模型")
args = vars(ap.parse_args())

MODELS = {                                                      # 定义了MODELS字典，将模型名称（字符串）映射到真实的Keras类
    "vgg16": VGG16,
    "vgg19": VGG19,
    "inception": InceptionV3,
    "xception": Xception,
    "resnet": ResNet50
}

if args["model"] not in MODELS.keys():                          # 如果字典中找不到 --model名称，将抛出Asserti
    raise AssertionError("The --model command line argument should"
                         "be a key in the 'MODELS' dictionary")

"""
1. 卷积神经网络将图像作为输入，然后返回与类标签相对应的一组概率作为输出
2. VGG16，VGG19和ResNet均接受224×224输入图像，而Inception V3和Xception需要299×299像素输入
"""
inputShape = (224,224)                                          # 初始化为224×224，如果使用Inception或Xception，需要把inputShape设为299×299像素
preprocess = imagenet_utils.preprocess_input                    # 执行平均减法，如果使用Inception或Xception，需要把preprocess_input设置为separate pre-processing function

"""
从磁盘加载预训练模型weight并实例化模型（如果第一次运行这个脚本，需要下载权重参数文件）
"""
print("[INFO] loading{}...".format(args["model"]))
Network = MODELS[args["model"]]                                 # 从--model命令行参数得到model的名字，通过MODEL字典映射到响应的类
model = Network(weights = "imagenet")                           # 使用预训练的ImageNet权重实例化卷积神经网络

"""准备图像进行分类"""
print("[INFO] loading and pre-processing image...")
image = load_img(args["image"],target_size=inputShape)          # 从磁盘加载图像，并且调整图片的宽度和高度
image = img_to_array(image)                                     # 将图像转化为Numpy数组，现在为（imputShape[0],inputShape[1],3）的Numpy数组
image = np.expand_dims(image,axis=0)                            # 添加一个额外的颜色通道（方便输入卷积神经网络），现在为（1，imputShape[0],inputShape[1],3）
image = preprocess(image)                                       # 预处理归一化

"""获取输出分类"""
print("[INFO]classifying image with'{}'...".format(args["model"]))
preds = model.predict(image)                                    # 调用CNN中的.predict得到预测结果
p = imagenet_utils.decode_predictions(preds)                    # 得到标签名字，提高可读性

for(i,(imagenetID,label,prob)) in enumerate(p[0]):
    print("{}. {}:{:.2f}%".format(i+1,label,prob*100))          # 将前5个预测输出到终端

"""通过OpenCV从磁盘加载我们的输入图像，在图像上绘制#1预测，最后将显示在我们的屏幕上"""
orig = cv2.imread(args["image"])
(imagenetID,label,prob) = P[0][0]
cv2.putText(orig,"Label:{},{:.2f}%".format(label,prob*100),
            (10,30),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,0,255),2)
cv2.imshow("Classification",orig)
cv2.waitKey(0)
