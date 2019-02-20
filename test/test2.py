import tensorflow as tf
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


session = tf.Session()
x = tf.placeholder(tf.float32,shape=[None,img_size,img_size,num_channels],name='x')

##labels
y_true = tf.placeholder(tf.float32,shape=[None,len(classes)],name='y_true')
y_true_cls = tf.argmax(y_true,dimension=1)

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

session.run(tf.global_variables_initializer())