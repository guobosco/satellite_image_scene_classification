import dataset
import tensorflow as tf
import time
import numpy as np
from numpy.random import seed
from tensorflow import set_random_seed
import matplotlib.pyplot as plt


#确定好一个随机的种子，确定好之后，之后的每一次随机都是相同的结果，保证验证集和训练集相同
seed(10)
set_random_seed(20)

batch_size = 32                                         #每次进入队列张数
classes =["agricultural","airplane","baseballdiamond",
          "beach","buildings","chaparral","denseresidential",
          "forest"]
num_classes = len(classes)


train_path = "/Users/mac/Documents/GitHub/DataSet/UC_Merge_LandUse/images_8"

validation_size = 0.2                                   #样本率
num_channels = 3                                        #输入图像的通道数
img_size = 64

print('<<<————开始准备数据集————>>>')
data = dataset.read_train_sets(train_path,img_size,classes,validation_size=validation_size)

print("训练集的数量：\t{}".format(len(data.train.labels)))
print("测试集的数量：\t{}".format(len(data.valid.labels)))

print("******数据准备完成******")

print("卷积网络结构初始化开始————>>>")
session = tf.Session()
x = tf.placeholder(tf.float32,shape=[None,img_size,img_size,num_channels],name='x')

##labels
y_true = tf.placeholder(tf.float32,shape=[None,len(classes)],name='y_true')
y_true_cls = tf.argmax(y_true,dimension=1)      #去值最大的那一个



#构造网络结构
filter_size_conv1 = 3#卷积核的大小
num_filters_conv1 = 32#得到32个特征图

filter_size_conv2 = 3
num_filters_conv2 = 32#32个特征图

filter_size_conv3 = 3
num_filters_conv3 = 64#64个特征图

fc_layer_size = 1024#映射成1024维的特征

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

    weights= create_weights(shape=[conv_filter_size,conv_filter_size,num_input_channels,num_filters])#3*3 3 32
    biases = create_biases(num_filters)

    layer=tf.nn.conv2d(input=input,
                       filter=weights,
                       strides=[1,1,1,1 ],
                       padding='SAME')
    layer +=biases

    layer = tf.nn.relu(layer)

    layer = tf.nn.max_pool(value=layer,
                           ksize=[1,2,2,1],
                           strides=[1,2,2,1],
                           padding='SAME')
    return layer
                                      #返回下一个batch

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
    msg="【遍数：{0}】\t【次数：{1}】:\t训练精度：{2:4.1%}\t泛化精度：{3:4.1%}\t泛化损失：{4:4.4}"


    print(msg.format(epoch+1,i,acc,val_acc,val_loss))
    print("当前时间为："+time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    return acc,val_acc
total_iterations = 0            #从第0次开始迭代

saver = tf.train.Saver()        #保存模型
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
        if i % int(data.train.num_examples/batch_size)==0:
            val_loss=session.run(cost,feed_dict=feed_dict_val)
            val_loss_list.append(val_loss)
            epoch = int(i/int(data.train.num_examples/batch_size))
            acc,val_acc = show_progress(epoch,feed_dict_tr,feed_dict_val,val_loss,i)
            acc_list.append(acc)
            val_acc_list.append(val_acc)
            saver.save(session,'/Users/mac/Documents/GitHub/models/2_21_class8/class.ckpt',global_step=i)

    total_iterations += num_iteration
print("马上开始10000次训练：(每次32张图片)")
acc_list = []
val_acc_list = []
val_loss_list = []
train(num_iteration=100)
acc_list = np.array(acc_list)
val_acc_list = np.array(val_acc_list)
val_loss_list = np.array(val_loss_list)
ax = np.arange(1,20)
plt.plot(ax,acc_list,color="blue",lw=2)
plt.plot(ax,val_acc_list,color="red",lw=2)
plt.xlabel("epoch")
plt.ylabel("acc")
plt.title("February_21_class8")
plt.savefig("/Users/mac/Documents/GitHub/models/2_21_class8/acc_2.png")
plt.show()