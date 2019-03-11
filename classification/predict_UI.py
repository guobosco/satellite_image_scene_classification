import os
from PyQt5 import QtCore
import tensorflow as tf
import numpy as np
import cv2

#0. 导入包和模块
from PyQt5.Qt import *

#1. 创建窗口类
class Uniformization_Window(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("第一个关键词预测")
        self.resize(1000,300)
        self.setup_ui()

    #设置窗口ui
    def setup_ui(self):
        # 1. 创建控件,例如：lable = QLabel(self)
        lable1 = QLabel(self)
        lable2 = QLabel(self)
        btn1 = QPushButton(self)
        lable3 = QLabel(self)
        lable4 = QLabel(self)

        btn5 = QPushButton(self)
        lable12 = QLabel(self)

        # 2. 设置控件
        lable1.move(20, 60)
        lable1.setText("请选择图片：")
        lable2.move(40, 95)
        lable2.setText("点击选文件：")
        btn1.move(120, 90)
        btn1.setText("浏览")

        lable3.setText("还未选择工作路径")
        lable3.setGeometry(QtCore.QRect(200, 95, 800, 20))


        btn5.setGeometry(QtCore.QRect(40, 180, 100, 50))
        btn5.setText("识别关键词")
        lable12.setText("结果")
        lable12.setGeometry(QtCore.QRect(160, 168, 440, 60))

        # 槽函数
        def in_path():
            path = QFileDialog.getOpenFileName(self, "选择一个文件", './')
            path = path[0]
            lable3.setText("当前工作路径：" + path)
            print(path)

        btn1.clicked.connect(in_path)


        def uniformization_function():
            # 源文件路径

            path = '/' + lable3.text()[8:]
            image_size = 64
            num_channels = 3
            images = []
            image = cv2.imread(path)
            image = cv2.resize(image, (image_size, image_size), 0, 0, cv2.INTER_LINEAR)
            images.append(image)
            images = np.array(images, dtype=np.uint8)
            images = np.multiply(images, 1.0 / 255.0)

            x_batch = images.reshape(1, image_size, image_size, num_channels)
            sess = tf.Session()

            saver = tf.train.import_meta_graph('/Users/mac/Documents/GitHub/models/summary_module/UC_Merge/class.ckpt-9776.meta')

            saver.restore(sess, '/Users/mac/Documents/GitHub/models/summary_module/UC_Merge/class.ckpt-9776')

            graph = tf.get_default_graph()

            y_pred = graph.get_tensor_by_name("y_pred:0")
            x = graph.get_tensor_by_name("x:0")
            y_ture = graph.get_tensor_by_name("y_true:0")
            y_test_images = np.zeros((1, 21))

            feed_dict_testing = {x: x_batch, y_ture: y_test_images}
            result = sess.run(y_pred, feed_dict=feed_dict_testing)
            res_label = ["农田", "机场", "棒球场",
                         "沙滩", "建筑", "丛林", "居民区",
                         "森林", "高速", "高尔夫球场", "码头", "十字路口",
                         "有绿化的居民区", "房车停车场", "立交桥", "停车场",
                         "河流", "飞机跑道", "独立小屋", "储物罐", "网球场"]

            print(res_label[result.argmax()])
            lable12.setStyleSheet("QLabel{font-size:50px}")
            lable12.setText(res_label[result.argmax()])
        btn5.clicked.connect(uniformization_function)

    #写功能函数
    def QObject_test(self):
        pass


#2.测试代码
if __name__ == '__main__':

    import sys

    #2.1 创建一个应用程序对象
    app = QApplication(sys.argv)

    #2.2 创建窗口
    window = Uniformization_Window()

    #2.3 展示窗口
    window.show()

    #2.4 应用程序的执行，进入到消息循环
    sys.exit(app.exec_())