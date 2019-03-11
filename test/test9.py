import tensorflow as tf
import keras
from keras import layers
import keras.datasets.mnist as mnist

(train_image,train_label),(test_image,test_label) = mnist.load_data()

#初始化模型
model = keras.Sequential()

#添加层，构建网络
model.add(layers.Dense(64,activation='relu',input_shape=(784,)))

model.add(layers.Dense(10,activation="softmax"))

model.summary()

#编译模型，添加优化函数

model.compile(optimizer='adam',
              loss='sparse_categoracal_crossentropy',
              metrics=['accuracy'])

#训练模型
model.fit(train_image,train_label,epochs=50,batch_size=512)

#评价模型
model.evaluate(test_image,test_label)

#预测
model.predict(test_image[:10])
