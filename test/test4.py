"""
线性模型例子，使用梯度下降法GradientDescentOptimizer最小化loss
"""

import numpy as np
import tensorflow as tf

#准备数据
data_x = np.linspace(0,10,30)
data_y = data_x*3 + 7 +np.random.normal(0,1,30)

#定义变量
x = tf.placeholder(tf.float32,shape=None)
y = tf.placeholder(tf.float32,shape=None)
w = tf.Variable(1.,name = "weights")
b = tf.Variable(0.,name="weights")
pred = tf.multiply(x,w) + b #预测值
total_loss = tf.reduce_sum(tf.squared_difference(y,pred)) #损失
learning_rate = 0.0001 #学习率
train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(total_loss)

#定义Session,初始化变量
sess = tf.Session()
sess.run(tf.global_variables_initializer())

#运行session
for i in range(10000):#运行10000次
    sess.run(train_op,feed_dict={x:data_x,y:data_y})
    if i%1000 == 0:
        print(sess.run([total_loss,w,b],feed_dict={x:data_x,y:data_y}))

#评估模型
y11 = 11*w + b
sess.run(y11)