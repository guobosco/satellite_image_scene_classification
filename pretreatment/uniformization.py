import os,cv2
import numpy as np
from PyQt5 import QtCore

#0. 导入包和模块
from PyQt5.Qt import *

#1. 创建窗口类
class Uniformization_Window(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("均值化")
        self.resize(500,400)
        self.setup_ui()

    #设置窗口ui
    def setup_ui(self):
        # 1. 创建控件,例如：lable = QLabel(self)
        lable1 = QLabel(self)
        lable2 = QLabel(self)
        btn1 = QPushButton(self)
        lable3 = QLabel(self)

        lable7 = QLabel(self)
        lable8 = QLabel(self)
        lable9 = QLabel(self)
        lable10 = QLabel(self)
        btn4 = QPushButton(self)
        lable11 = QLabel(self)
        btn5 = QPushButton(self)
        lable12 = QLabel(self)
        progressBar = QProgressBar(self)

        # 2. 设置控件
        lable1.move(20, 50)
        lable1.setText("设置工作路径：")
        lable2.move(40, 85)
        lable2.setText("选择文件夹：")
        btn1.move(120, 80)
        btn1.setText("浏览")
        lable3.setText("还未选择工作路径")
        lable3.setGeometry(QtCore.QRect(200, 85, 800, 20))

        lable7.setGeometry(QtCore.QRect(360, 75, 60, 20))
        lable8.setGeometry(QtCore.QRect(360, 110, 60, 20))
        lable9.move(20, 150)
        lable9.setText("设置输入路径：")
        lable10.move(40, 185)
        lable10.setText("选择文件夹：")
        btn4.move(120, 180)
        btn4.setText("浏览")
        lable11.setText("还未选择输出路径")
        lable11.setGeometry(QtCore.QRect(200, 185, 800, 20))
        btn5.setGeometry(QtCore.QRect(40, 230, 100, 50))
        btn5.setText("开始")
        lable12.setText("进度")
        lable12.setGeometry(QtCore.QRect(160, 242, 40, 20))
        progressBar.setGeometry(QtCore.QRect(205, 242, 200, 20))

        # 槽函数
        def in_path():
            path = QFileDialog.getExistingDirectory(self, "选择一个文件夹", './')
            lable3.setText("当前工作路径：" + path)

        btn1.clicked.connect(in_path)

        # 槽函数
        def out_path():
            path = QFileDialog.getExistingDirectory(self, "选择一个文件夹", './')
            lable11.setText("当前工作路径：" + path)

        btn4.clicked.connect(out_path)

        def uniformization_function():
            lable12.setText("工作中")
            # 源文件路径
            classes = ["agricultural", "airplane", "baseballdiamond",
                       "beach", "buildings", "chaparral", "denseresidential",
                       "forest", "freeway", "golfcourse", "harbor", "intersection",
                       "mediumresidential", "mobilehomepark", "overpass", "parkinglot",
                       "river", "runway", "sparseresidential", "storagetanks", "tenniscourt"]
            read_path = '/' + lable3.text()[8:]
            # 写入路径
            write_path = '/' + lable11.text()[8:]
            """核心代码"""
            for fields in classes:
                path = os.path.join(read_path, fields)
                w_path = os.path.join(write_path, fields)
                file_names = os.listdir(path)
                for file_name in file_names:
                    img = cv2.imread(os.path.join(path, file_name), 1)
                    print(os.path.join(path, file_name))
                    per_image_Bmean = []
                    per_image_Gmean = []
                    per_image_Rmean = []
                    per_image_Bmean.append(np.mean(img[:, :, 0]))
                    per_image_Gmean.append(np.mean(img[:, :, 1]))
                    per_image_Rmean.append(np.mean(img[:, :, 2]))
                    B_mean = np.mean(per_image_Bmean)
                    G_mean = np.mean(per_image_Gmean)
                    R_mean = np.mean(per_image_Rmean)

                    height = img.shape[0]
                    width = img.shape[1]
                    for row in range(height):
                        for col in range(width):
                            B = img[row, col, 0]
                            G = img[row, col, 1]
                            R = img[row, col, 2]
                            img[row, col, 0] = B - B_mean
                            img[row, col, 1] = G - G_mean
                            img[row, col, 2] = R - R_mean
                    print(os.path.join(w_path, file_name))
                    cv2.imwrite(os.path.join(w_path, file_name), img)
                progressBar.setRange(0, 100)
                progressBar.setValue(100)
                lable12.setText("完成")
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