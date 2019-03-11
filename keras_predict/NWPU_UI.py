from PIL import Image
from keras.preprocessing import image

from PyQt5 import QtCore
import tensorflow as tf
import numpy as np
import sys
from PyQt5 import QtWidgets, QtGui

#0. 导入包和模块
from PyQt5.Qt import *

#1. 创建窗口类
class Uniformization_Window(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("第一个关键词预测")
        self.resize(800,350)
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
        lable1.move(20, 20)
        lable1.setText("请选择图片：")
        lable2.move(40, 55)
        lable2.setText("点击选文件：")
        btn1.move(120, 50)
        btn1.setText("浏览")

        lable3.setText("还未选择工作路径")
        lable3.setGeometry(QtCore.QRect(200, 55, 800, 20))
        lable4.setText("")
        lable4.setGeometry(QtCore.QRect(500, 80, 240, 240))

        btn5.setGeometry(QtCore.QRect(40, 180, 100, 50))
        btn5.setText("识别关键词")
        lable12.setText("结果")
        lable12.setGeometry(QtCore.QRect(160, 168, 440, 60))

        # 槽函数
        def in_path():
            path = QFileDialog.getOpenFileName(self, "选择一个文件", './')
            path = path[0]
            lable3.setText("当前图片路径：" + path)
            print(path)
            png = QtGui.QPixmap(path)
            lable4.setPixmap(png)

        btn1.clicked.connect(in_path)


        def uniformization_function():
            # 源文件路径

            IMG_W, IMG_H, IMG_CH = 150, 150, 3  # 单张图片的大小
            image_path = '/' + lable3.text()[8:]

            def predict(model, img_path, target_size):
                img = Image.open(img_path)  # 加载图片
                if img.size != target_size:
                    img = img.resize(target_size)

                x = image.img_to_array(img)
                x *= 1. / 255  # 相当于ImageDataGenerator(rescale=1. / 255)
                x = np.expand_dims(x, axis=0)  # 调整图片维度
                preds = model.predict(x)  # 预测
                return preds[0]

            from keras.models import load_model
            saved_model = load_model('/Users/mac/Documents/GitHub/models/summary_module/NWPU/self_design_NWPU45.h5')
            result = predict(saved_model, image_path, (IMG_W, IMG_H))
            res_label = ["飞机", "机场", "棒球场",
                 "篮球场", "沙滩", "桥梁", "丛林",
                 "教堂", "圆形农场", "云朵", "商业区", "密集居民区",
                 "沙漠", "森林", "高速公路", "高尔夫球场","田径场",
                 "码头", "工业区", "十字路口", "岛屿", "湖泊",
                 "草地", "中等住宅", "房车停车场", "山脉", "立交桥", "宫殿", "停车场",
                 "高速公路", '火车站', "方形农田", "河流", "环形转盘", "跑道", "海上冰山", "舰船",
                 "冰山", "独立房屋", "体育场", "储罐", "网球场", "梯田", "热电站", "湿地"]

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