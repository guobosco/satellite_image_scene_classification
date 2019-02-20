import tensorflow as tf
import glob
import numpy as np
import random

train_path = "/Users/mac/Documents/GitHub/satellite_image_scene_classification/classification/images_uni_224"
classes =["agricultural","airplane","baseballdiamond",
          "beach","buildings","chaparral","denseresidential",
          "forest","freeway","golfcourse","harbor","intersection",
          "mediumresidential","mobilehomepark","overpass","parkinglot",
          "river","runway","sparseresidential","storagetanks","tenniscourt"]
image_filenames = glob.glob(train_path + '/*/*.tif', recursive=True)            #使用glob读取所有的文件
random.shuffle(image_filenames)                                                 #对文件名进行乱序处理
lables = list(map(lambda x:x.split('/')[-1].split('.')[0][:-2],image_filenames))#获取标签

def lables_floater(lable):
    if lable == classes[0]:
        lable = [1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
    elif lable == classes[1]:
        lable = [0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
    elif lable == classes[2]:
        lable = [0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
    elif lable == classes[3]:
        lable = [0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
    elif lable == classes[4]:
        lable = [0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
    elif lable == classes[5]:
        lable = [0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
    elif lable == classes[6]:
        lable = [0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
    elif lable == classes[7]:
        lable = [0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
    elif lable == classes[8]:
        lable = [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
    elif lable == classes[9]:
        lable = [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
    elif lable == classes[10]:
        lable = [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
    elif lable == classes[11]:
        lable = [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
    elif lable == classes[12]:
        lable = [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
    elif lable == classes[13]:
        lable = [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
    elif lable == classes[14]:
        lable = [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0]
    elif lable == classes[15]:
        lable = [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0]
    elif lable == classes[16]:
        lable = [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0]
    elif lable == classes[17]:
        lable = [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0]
    elif lable == classes[18]:
        lable = [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0]
    elif lable == classes[19]:
        lable = [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0]
    elif lable == classes[20]:
        lable = [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0]
    return lable
float_lables = list(map(lables_floater,lables))                                     #把标签转化为浮点类型
#image_filenames = list(map(lambda x:x.strip('.tif')+'.jpeg',image_filenames))
dataset = tf.data.Dataset.from_tensor_slices((image_filenames,float_lables))        #创建dataset

def pre_read(img_filename,lable):
    image = tf.read_file(img_filename)
    image = tf.image.decode_jpeg(image,channels=3,)
    image = tf.reshape(image,[224,224,3])
    image = tf.cast(image,tf.float32)
    lable = tf.reshape(lable,[21])
    return image,lable

dataset = dataset.map(pre_read)                                                     #读入图片
dataset = dataset.shuffle(100)                                                      #缓存100张
dataset = dataset.repeat(-1)
dataset = dataset.batch(64)

iterator = dataset.make_one_shot_iterator()                                         #创建迭代器
image_batch,lable_batch = iterator.get_next()                                       #返回下一个batch

'''构建CNN网络'''
conv2d_1 = tf.contrib.layers.convolution2d(
    image_batch,
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
pool3_flat = tf.reshape(pool_3, [-1, 25*25*64])

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
pred = tf.cast(tf.argmax(tf.nn.softmax(comb_out),1),tf.int32)
loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(labels=lable_batch, logits=comb_out))
learning_rate = 0.0001
train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)

#accuracy = tf.reduce_mean(tf.cast(tf.equal(pred,lable_batch), tf.float32))

saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for step in range(3000):
        sess.run(train_step)

        if(step%10 == 0):
            res = sess.run([loss])
            print(step,res)
            print(pred)
            print('******')
            saver.save(sess,'./model',global_step= step)