import tensorflow as tf
import numpy as np

x = tf.placeholder("float", shape=[None,32,32,3])
y_ = tf.placeholder(tf.int32, shape=[None])
keep_prob = tf.placeholder("float")

def im_pre(image):
    new_img = tf.image.random_brightness(image, max_delta=63) #随机调节图像的亮度
    new_img = tf.image.random_flip_left_right(new_img) #随机地左右翻转图像
    new_img = tf.image.random_contrast(new_img, lower=0.2, upper=1.8) #随机地调整图像对比度
    new_img = tf.image.per_image_standardization(image)
    return new_img

img =  tf.map_fn(im_pre,x)
#定义batch的大小
b_size = 128

#定义学习速率
learning_rate = 0.0001

#定义正则化函数
reg = tf.contrib.layers.l2_regularizer(scale=0.1)

#三层卷积网络
conv2d_1 = tf.contrib.layers.convolution2d(
    img,
    num_outputs=32,
    weights_initializer=tf.truncated_normal_initializer(stddev=0.0001),
    weights_regularizer = reg,
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
    weights_regularizer = reg,
    kernel_size = (5,5),
    activation_fn = tf.nn.relu,
    stride = (1,1),
    padding='SAME',
    trainable = True)
pool_2 = tf.nn.max_pool(conv2d_2,
                       ksize = [1,3,3,1],
                       strides = [1,2,2,1],
                       padding='SAME')

conv_2d_w3 = tf.Variable(tf.truncated_normal([5, 5, 32, 64],stddev=0.01))
conv_2d_b3 = tf.Variable(tf.truncated_normal([64]))
conv2d_3 = tf.nn.conv2d(pool_2, conv_2d_w3,strides=[1, 1, 1, 1], padding='SAME') + conv_2d_b3

conv2d_3_output = tf.nn.relu(conv2d_3)

pool_3 = tf.nn.max_pool(conv2d_3_output,
                       ksize = [1,3,3,1],
                       strides = [1,2,2,1],
                       padding='SAME')

pool3_flat = tf.reshape(pool_3, [-1, 4*4*64])
fc_1 = tf.contrib.layers.fully_connected(
                        pool3_flat,
                        1024,
                        weights_initializer=tf.truncated_normal_initializer(stddev=0.1),
                        weights_regularizer = reg,
                        activation_fn = tf.nn.relu)

fc_2 = tf.contrib.layers.fully_connected(
                        fc_1,
                        128,
                        weights_initializer=tf.truncated_normal_initializer(stddev=0.1),
                        weights_regularizer = reg,
                        activation_fn = tf.nn.relu)
fc2_drop = tf.nn.dropout(fc_2, keep_prob)

out_w1 = tf.Variable(tf.truncated_normal([128,10]))
out_b1 = tf.Variable(tf.truncated_normal([10]))
combine = tf.matmul(fc2_drop,out_w1)+out_b1
#softmax分类函数
pred = tf.cast(tf.argmax(tf.nn.softmax(combine),1),tf.int32)

weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

reg_ws = tf.contrib.layers.apply_regularization(reg,weights_list = weights)

loss = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_, logits=combine))

loss_fn = loss + tf.reduce_sum(reg_ws)

train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss_fn)

accuracy = tf.reduce_mean(tf.cast(tf.equal(pred,y_), tf.float32))

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return

data_list = []
label_list = []
for i in range(1,6):
    data = unpickle('/Users/mac/Documents/GitHub/satellite_image_scene_classification/test/cifar-10-batches-py/data_batch_{}'.format(i))
    data_list.append(data[b'data'])
    label_list.append(data[b'labels'])

all_data = np.concatenate(data_list)
all_label = np.concatenate(label_list)

def generatebatch(X,Y,n_examples, batch_size):
    for batch_i in range(n_examples // batch_size):
        start = batch_i*batch_size
        end = start + batch_size
        batch_xs = X[start:end]
        batch_ys = Y[start:end]
        yield batch_xs, batch_ys # 生成每一个batch

saver = tf.train.Saver()
sess = tf.Session()
sess.run(tf.global_variables_initializer())
ii = 0
for epoch in range(0,100):
    index = np.random.permutation(all_label.shape[0])
    all_data = all_data[index]
    all_label = all_label[index]
    for batch_xs,batch_ys in generatebatch(all_data,all_label,all_label.shape[0],b_size):
        batch_xs = np.array(list(map(lambda x:x.reshape([3,1024]).T.reshape([32,32,3]),batch_xs)))
        sess.run(train_step,feed_dict={x: batch_xs, y_: batch_ys,keep_prob: 0.5})
        if ii%100 == 0:
            print(sess.run([loss,accuracy,],feed_dict={x: batch_xs, y_: batch_ys,keep_prob: 1}))
        ii += 1
    if(epoch%2==0):
        res = sess.run([loss,accuracy],feed_dict={x: batch_xs, y_: batch_ys,keep_prob: 1})
        print(epoch,res)
        saver.save(sess,'./lesson20',global_step = epoch)

saver.save(sess,'./lesson10',global_step=50000)

test = unpickle('./cifar-10/test_batch')

test_label_hot = test[b'labels']

test_data = test[b'data']

righ = []
for batch_xs,batch_ys in generatebatch(test_data,test_label_hot,test_data.shape[0],128):
    batch_xs = np.array(list(map(lambda x:x.reshape([3,1024]).T.reshape([32,32,3]),batch_xs)))
    acc = sess.run(accuracy,feed_dict={x: batch_xs, y_: batch_ys,keep_prob: 1})
    righ.append(acc)
print(sess.run(tf.reduce_mean(righ)))

initial_step = 0

import os

import os
ckpt = tf.train.get_checkpoint_state(os.path.dirname('__file__'))
ckpt.model_checkpoint_path = 'lesson20-90'

saver = tf.train.Saver()
sess = tf.Session()
sess.run(tf.global_variables_initializer())
saver.restore(sess,ckpt.model_checkpoint_path)

ii = 0
for epoch in range(49,400):
    index = np.random.permutation(all_label.shape[0])
    all_data = all_data[index]
    all_label = all_label[index]
    for batch_xs,batch_ys in generatebatch(all_data,all_label,all_label.shape[0],b_size):
        batch_xs = batch_xs.reshape((b_size,32,32,3))
        sess.run(train_step,feed_dict={x: batch_xs, y_: batch_ys,keep_prob: 0.5})
        if ii%100 == 0:
            print(sess.run([loss,accuracy,],feed_dict={x: batch_xs, y_: batch_ys,keep_prob: 1}))
        ii += 1
    if(epoch%2==0):
        res = sess.run([loss,accuracy],feed_dict={x: batch_xs, y_: batch_ys,keep_prob: 1})
        print(epoch,res)
        saver.save(sess,'./lesson9',global_step = epoch)

sess.run(tf.reduce_sum(reg_ws))