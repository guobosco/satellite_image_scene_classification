from PyQt5 import QtCore
#0. 导入包和模块
from PyQt5.Qt import *
from PIL import Image
from keras.preprocessing import image
import glob
import numpy as np
from collections import Counter
from PyQt5 import QtWidgets, QtGui
#1. 创建窗口类
class Uniformization_Window(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("分割检测获得词袋")
        self.resize(1200,580)
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
        lable4.setGeometry(QtCore.QRect(500, 80, 640, 480))

        btn5.setGeometry(QtCore.QRect(40, 180, 100, 50))
        btn5.setText("得到词袋")
        lable12.setText("结果")
        lable12.setGeometry(QtCore.QRect(160, 168, 440, 60))

        # 槽函数
        def in_path():
            path = QFileDialog.getExistingDirectory(self, "选择一个文件夹", './')
            path = path
            lable3.setText("当前图片路径：" + path)
            print(path)

        btn1.clicked.connect(in_path)


        def uniformization_function():
            # 源文件路径

            IMG_W, IMG_H, IMG_CH = 150, 150, 3  # 单张图片的大小
            image_path = '/' + lable3.text()[8:]
            sigle_image_path = glob.glob(image_path + '/*.jpg', recursive=True)
            res_label = ["飞机", "机场", "棒球场",
                         "篮球场", "沙滩", "桥梁", "丛林",
                         "教堂", "圆形农场", "云朵", "商业区", "密集居民区",
                         "沙漠", "森林", "高速公路", "高尔夫球场", "田径场",
                         "码头", "工业区", "十字路口", "岛屿", "湖泊",
                         "草地", "中等住宅", "房车停车场", "方形农田", "立交桥", "宫殿", "停车场",
                         "高速公路", '火车站', "方形农田", "中等住宅", "环形转盘", "跑道", "海上冰山", "舰船",
                         "冰山", "独立房屋", "体育场", "储罐", "网球场", "梯田", "热电站", "湿地"]

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
            res = []
            for imag in sigle_image_path:
                result = predict(saved_model, imag, (IMG_W, IMG_H))
                lable_res = res_label[result.argmax()]
                res.append(lable_res)
                print(lable_res)
            res_f = Counter(res)
            res_f = res_f.most_common()
            print(res_f)
            import matplotlib.pyplot as plt
            import matplotlib
            matplotlib.rcParams['font.sans-serif'] = ['SimHei']
            matplotlib.rcParams['axes.unicode_minus'] = False
            count = [res_f[0][1], res_f[1][1], res_f[2][1], res_f[3][1], res_f[4][1], res_f[5][1], res_f[6][1],
                     res_f[7][1]]
            plt.barh(range(8), count, height=0.7, color='blue', alpha=0.8)
            plt.yticks(range(8),
                       [res_f[0][0], res_f[1][0], res_f[2][0], res_f[3][0], res_f[4][0], res_f[5][0], res_f[6][0],
                        res_f[7][0]])
            plt.xlim(0, 200)
            plt.xlabel("次数")
            plt.title("分割图片类别统计")
            for x, y in enumerate(count):
                plt.text(y + 0.2, x - 0.1, '%s' % y)
            plt.show()
            plt.savefig("/Users/mac/Desktop/zhifangtu/fake.png")
            png = QtGui.QPixmap("/Users/mac/Desktop/zhifangtu/fake2.jpeg")
            lable4.setPixmap(png)
            lable12.setStyleSheet("QLabel{font-size:36px}")
            lable12.setText("词袋统计直方图：")
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