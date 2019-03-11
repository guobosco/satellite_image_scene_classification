import tensorflow as tf
import numpy as np
import cv2

image_size = 64
num_channels = 3
images = []

path = "/Users/mac/Documents/GitHub/DataSet/UC_Merge_LandUse/images_uni_224/chaparral/chaparral05.tif"
image = cv2.imread(path)

image = cv2.resize(image,(image_size,image_size),0,0,cv2.INTER_LINEAR)
images.append(image)
images = np.array(images,dtype=np.uint8)
images = np.multiply(images,1.0/255.0)

x_batch = images.reshape(1,image_size,image_size,num_channels)
sess = tf.Session()

saver = tf.train.import_meta_graph('/Users/mac/Documents/GitHub/models/2_21/class.ckpt-9776.meta')

saver.restore(sess,'/Users/mac/Documents/GitHub/models/2_21/class.ckpt-9776')

graph = tf.get_default_graph()

y_pred = graph.get_tensor_by_name("y_pred:0")
x = graph.get_tensor_by_name("x:0")
y_ture = graph.get_tensor_by_name("y_true:0")
y_test_images = np.zeros((1,21))

feed_dict_testing = {x:x_batch,y_ture:y_test_images}
result = sess.run(y_pred,feed_dict=feed_dict_testing)
res_label = ["农田", "机场", "棒球场",
                       "沙滩", "建筑", "丛林", "居民区",
                       "森林", "高速", "高尔夫球场", "码头", "十字路口",
                       "有绿化的居民区", "房车停车场", "立交桥", "停车场",
                       "河流", "飞机跑道", "独立小屋", "储物罐", "网球场"]

print(res_label[result.argmax()])
