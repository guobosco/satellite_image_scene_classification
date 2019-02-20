import dataset
import tensorflow as tf
import time
from datetime import timedelta
import math
import random
import numpy as np
import cv2
from numpy.random import seed
from tensorflow import set_random_seed

#确定好一个随机的种子
seed(10)
set_random_seed(20)

img_size = 224
batch_size = 50 #每次进入队列张数
classes =["agricultural","airplane","baseballdiamond",
          "beach","buildings","chaparral","denseresidential",
          "forest","freeway","golfcourse","harbor","intersection",
          "mediumresidential","mobilehomepark","overpass","parkinglot",
          "river","runway","sparseresidential","storagetanks","tenniscourt"]
validation_size = 0.2#样本率
num_channels = 3#输入图像的通道数
train_path = "/Users/mac/Documents/GitHub/satellite_image_scene_classification/classification/images_uni_224"
print('<<<————开始准备数据集————>>>')
data = dataset.read_train_sets(train_path,classes,validation_size=validation_size)

print("训练集的数量：\t{}".format(len(data.train.labels)))
print("测试集的数量：\t{}".format(len(data.valid.labels)))

print("******数据准备完成******")

print("卷积网络结构初始化开始————>>>")
session = tf.Session()
x = tf.placeholder(tf.float32,shape=[None,img_size,img_size,num_channels],name='x')

##labels
y_true = tf.placeholder(tf.float32,shape=[None,len(classes)],name='y_true')
y_true_cls = tf.argmax(y_true,dimension=1)



#构造网络结构
filter_size_conv1 = 3
num_filters_conv1 = 32

filter_size_conv2 = 3
num_filters_conv2 = 32

filter_size_conv3 = 3
num_filters_conv3 = 64

fc_layer_size = 1024

#构造权重参数
def create_weights(shape):
    return tf.Variable(tf.truncated_normal(shape,stddev=0.05))
#构造biases参数
def create_biases(size):
    return tf.Variable(tf.constant(0.05,shape=[size]))
def create_convolutional_layer(input,
                               num_input_channels,
                               conv_filter_size,
                               num_filters):
    weights= create_weights(shape=[conv_filter_size,conv_filter_size,num_input_channels,num_filters])
    biases = create_biases(num_filters)

    layer=tf.nn.conv2d(input=input,
                       filter=weights,
                       strides=[1,1,1,1 ],
                       padding='SAME')
    layer+=biases

    layer=tf.nn.relu(layer)

    layer = tf.nn.max_pool(value=layer,
                           ksize=[1,2,2,1],
                           strides=[1,2,2,1],
                           padding='SAME')

    return layer

def create_flatten_layer(layer):
    layer_shape = layer.get_shape()
    num_features = layer_shape[1:4].num_elements()
    layer=tf.reshape(layer,[-1,num_features])

    return layer

def create_fc_layer(input,
                    num_inputs,
                    num_outputs,
                    use_relu=True):
    weights = create_weights(shape=[num_inputs,num_outputs])
    biases = create_biases(num_outputs)

    layer = tf.matmul(input,weights)+biases
    layer=tf.nn.dropout(layer,keep_prob=0.7)

    if use_relu:
        layer = tf.nn.relu(layer)
    return layer





layer_conv1= create_convolutional_layer(input=x,
                                        num_input_channels=num_channels,
                                        conv_filter_size=filter_size_conv1,
                                        num_filters=num_filters_conv1)

layer_conv2= create_convolutional_layer(input=layer_conv1,
                                        num_input_channels=num_filters_conv1,
                                        conv_filter_size=filter_size_conv2,
                                        num_filters=num_filters_conv2)

layer_conv3= create_convolutional_layer(input=layer_conv2,
                                        num_input_channels=num_filters_conv2,
                                        conv_filter_size=filter_size_conv3,
                                        num_filters=num_filters_conv3)

#把第三个卷积层出来的特征拉长
layer_flat = create_flatten_layer(layer_conv3)

layer_fc1 = create_fc_layer(input=layer_flat,
                            num_inputs=layer_flat.get_shape()[1:4].num_elements(),
                            num_outputs=fc_layer_size,
                            use_relu=True)

layer_fc2 = create_fc_layer(input=layer_fc1,
                            num_inputs=fc_layer_size,
                            num_outputs=len(classes),
                            use_relu=False)
print("******卷积网络结构初始化完成******")

y_pred= tf.nn.softmax(layer_fc2,name='y_pred')

y_pred_cls=tf.argmax(y_pred,dimension=1)
session.run(tf.global_variables_initializer())
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=layer_fc2,
                                                        labels=y_true)
cost = tf.reduce_mean(cross_entropy)

#学习率
optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)
#
correct_prediction = tf.equal(y_pred_cls,y_true_cls)
accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

session.run(tf.global_variables_initializer())

def show_progress(epoch,feed_dict_train,feed_dict_validate,val_loss,i):
    acc = session.run(accuracy,feed_dict=feed_dict_train)
    val_acc=session.run(accuracy,feed_dict=feed_dict_validate)
    msg="【轮次：{0}】\t【次数：{1}】:\t训练精度：{2:4.1%}\t泛化精度：{3:4.1%}\t损失：{4:4.4}"
    print(msg.format(epoch+1,i,acc,val_acc,val_loss))
    print("当前时间为："+time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    print("下50张图片训练开始——————————>>>")

total_iterations = 0

saver = tf.train.Saver()
def train(num_iteration):
    global total_iterations

    for i in range(total_iterations,
                   total_iterations + num_iteration):
        x_batch,y_true_batch,cls_batch = data.train.next_batch(batch_size)
        x_valid_batch,y_valid_batch,valid_cls_batch=data.valid.next_batch(batch_size)

        feed_dict_tr = {x:x_batch,
                        y_true:y_true_batch}
        feed_dict_val = {x:x_valid_batch,
                         y_true:y_valid_batch}

        session.run(optimizer,feed_dict=feed_dict_tr)
        #writer = tf.summary.FileWriter('/Users/mac/Desktop/demo2', session.graph)
        #if i % int(data.train.num_examples/batch_size)==0:
        val_loss=session.run(cost,feed_dict=feed_dict_val)
        epoch = int(i/int(data.train.num_examples/batch_size))

        show_progress(epoch,feed_dict_tr,feed_dict_val,val_loss,i)
        if i % (10*int(data.train.num_examples / batch_size)) == 0:
            saver.save(session,'/Users/mac/Documents/GitHub/satellite_image_scene_classification/classification/models/class.ckpt',global_step=i)

    total_iterations += num_iteration
print("马上开始5000次训练：(每次50张图片)")
train(num_iteration=5000)
