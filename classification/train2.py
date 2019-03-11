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
          "forest","freeway","golfcourse","harbor","intersection",
          "mediumresidential","mobilehomepark","overpass","parkinglot",
          "river","runway","sparseresidential","storagetanks","tenniscourt"]
num_classes = len(classes)


train_path = "/Users/mac/Documents/GitHub/DataSet/UC_Merge_LandUse/images_uni_224"

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

'''构建CNN网络'''
conv2d_1 = tf.contrib.layers.convolution2d(
    x,
    num_outputs=32,
    weights_initializer=tf.truncated_normal_initializer(stddev=0.001),
    kernel_size = (5,5),
    activation_fn = tf.nn.relu,
    stride = (1,1),
    padding='SAME',
    trainable = True)
pool_1 = tf.nn.max_pool(conv2d_1,
                       ksize = [1,3,3,1],
                       strides = [1,2,2,1],
                       padding='SAME')

conv2d_2 = tf.contrib.layers.convolution2d(
    pool_1,
    num_outputs=32,
    weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
    kernel_size = (5,5),
    activation_fn = tf.nn.relu,
    stride = (1,1),
    padding='SAME',
    trainable = True)
pool_2 = tf.nn.max_pool(conv2d_2,
                       ksize = [1,3,3,1],
                       strides = [1,2,2,1],
                       padding='SAME')

conv2d_3 = tf.contrib.layers.convolution2d(
    pool_2,
    num_outputs=64,
    weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
    kernel_size = (5,5),
    activation_fn = tf.nn.relu,
    stride = (1,1),
    padding='SAME',
    trainable = True)
pool_3 = tf.nn.max_pool(conv2d_3,
                       ksize = [1,3,3,1],
                       strides = [1,2,2,1],
                       padding='SAME')
pool3_flat = tf.reshape(pool_3, [-1, 131072])

fc_1 = tf.contrib.layers.fully_connected(
                        pool3_flat,
                        1024,
                        weights_initializer=tf.truncated_normal_initializer(stddev=0.1),
                        activation_fn = tf.nn.relu)
fc_2 = tf.contrib.layers.fully_connected(
                        fc_1,
                        192,
                        weights_initializer=tf.truncated_normal_initializer(stddev=0.1),
                        activation_fn = tf.nn.relu)
out_w1 = tf.Variable(tf.truncated_normal([192,21]))
out_b1 = tf.Variable(tf.truncated_normal([21]))
comb_out = tf.matmul(fc_2,out_w1)+out_b1

print("******卷积网络结构初始化完成******")

y_pred= tf.nn.softmax(comb_out,name='y_pred')
y_pred_cls=tf.argmax(y_pred,dimension=1)

session.run(tf.global_variables_initializer())

cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=comb_out,
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
            val_loss = session.run(cost, feed_dict=feed_dict_val)
            val_loss_list.append(val_loss)
            epoch = int(i / int(data.train.num_examples / batch_size))
            acc, val_acc = show_progress(epoch, feed_dict_tr, feed_dict_val, val_loss, i)
            acc_list.append(acc)
            val_acc_list.append(val_acc)
            #saver.save(session,'/Users/mac/Documents/GitHub/models/2_20/class.ckpt',global_step=i)

    total_iterations += num_iteration
print("马上开始10000次训练：(每次32张图片)")
acc_list = []
val_acc_list = []
val_loss_list = []
train(num_iteration=1000)
acc_list = np.array(acc_list)
val_acc_list = np.array(val_acc_list)
val_loss_list = np.array(val_loss_list)
ax = np.arange(1,21)
plt.plot(ax,acc_list,color="blue",lw=2)
plt.plot(ax,val_acc_list,color="red",lw=2)
plt.xlabel("epoch")
plt.ylabel("acc")
plt.title("February_21_class21_newCNN")
plt.savefig("/Users/mac/Documents/GitHub/models/2_21_newCNN/acc_21.png")
plt.show()
