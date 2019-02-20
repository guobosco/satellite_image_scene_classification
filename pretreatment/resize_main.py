#0. 导入包和模块
from PyQt5.Qt import *
import sys,cv2,os
from PyQt5 import QtCore

#1. 创建窗口类
class Resize_Window(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("改变输入图片的大小")
        self.resize(500,400)
        self.setup_ui()

    #设置窗口ui
    def setup_ui(self):
        #1. 创建控件,例如：lable = QLabel(self)
        lable1 = QLabel(self)
        lable2 = QLabel(self)
        btn1 = QPushButton(self)
        lable3 = QLabel(self)
        lable4 = QLabel(self)
        lable5 = QLabel(self)
        lable6 = QLabel(self)
        lineEdit1 = QLineEdit("224",self)
        lineEdit2 = QLineEdit("224",self)
        comboBox1 = QComboBox(self)
        comboBox2 = QComboBox(self)
        btn2 = QPushButton(self)
        btn3 = QPushButton(self)
        lable7 = QLabel(self)
        lable8 = QLabel(self)
        lable9 = QLabel(self)
        lable10 = QLabel(self)
        btn4 = QPushButton(self)
        lable11 = QLabel(self)
        btn5 = QPushButton(self)
        lable12 = QLabel(self)
        progressBar = QProgressBar(self)

        #2. 设置控件
        lable1.move(20,20)
        lable1.setText("设置输入路径：")
        lable2.move(40,55)
        lable2.setText("选择文件夹：")
        btn1.move(120,50)
        btn1.setText("浏览")
        lable3.setText("还未选择工作路径")
        lable3.setGeometry(QtCore.QRect(200, 55, 800, 20))
        lable4.move(20, 90)
        lable4.setText("设置输出像素：")
        lable5.setText("纵向(y)")
        lable5.setGeometry(QtCore.QRect(40, 125, 60, 20))
        lable6.setText("横向(x)")
        lable6.setGeometry(QtCore.QRect(40, 160, 60, 20))
        lineEdit1.setGeometry(QtCore.QRect(100, 125, 100, 20))
        lineEdit2.setGeometry(QtCore.QRect(100, 160, 100, 20))
        comboBox1.setGeometry(QtCore.QRect(200, 125, 80, 20))
        comboBox2.setGeometry(QtCore.QRect(200, 160, 80, 20))
        comboBox1.addItem("像素")
        comboBox2.addItem("像素")
        btn2.move(280, 120)
        btn2.setText("确定")
        btn3.move(280, 155)
        btn3.setText("确定")
        lable7.setGeometry(QtCore.QRect(360, 125, 60, 20))
        lable8.setGeometry(QtCore.QRect(360, 160, 60, 20))
        lable9.move(20, 200)
        lable9.setText("设置输入路径：")
        lable10.move(40, 235)
        lable10.setText("选择文件夹：")
        btn4.move(120, 230)
        btn4.setText("浏览")
        lable11.setText("还未选择输出路径") 
        lable11.setGeometry(QtCore.QRect(200, 235, 800, 20))
        btn5.setGeometry(QtCore.QRect(40, 280, 100, 50))
        btn5.setText("开始")
        lable12.setText("进度")
        lable12.setGeometry(QtCore.QRect(160, 292, 40, 20))
        progressBar.setGeometry(QtCore.QRect(205, 292, 200, 20))

        def resize_function():
            classes = ["agricultural", "airplane", "baseballdiamond",
                       "beach", "buildings", "chaparral", "denseresidential",
                       "forest", "freeway", "golfcourse", "harbor", "intersection",
                       "mediumresidential", "mobilehomepark", "overpass", "parkinglot",
                       "river", "runway", "sparseresidential", "storagetanks", "tenniscourt"]
            # 源文件路径
            read_path = '/'+lable3.text()[8:]
            # 写入路径
            write_path = '/'+lable11.text()[8:]
            # 设置像素大小
            pxy = int(lable7.text())
            pxx = int(lable8.text())
            """核心代码"""
            for fields in classes:
                path = os.path.join(read_path, fields)
                w_path = os.path.join(write_path, fields)
                file_names = os.listdir(path)
                for file_name in file_names:
                    img = cv2.imread(os.path.join(path, file_name), 1)
                    print(os.path.join(path, file_name))
                    if file_name[-1] == 'f':  # 确保是以.tif结尾的图像文件
                        img256 = cv2.imread(path + "/" + file_name)
                        img224 = cv2.resize(img256, (pxy, pxx))  # 本程序核心代码
                        cv2.imwrite(os.path.join(w_path, file_name), img224)
                        print(os.path.join(w_path, file_name))

        btn5.clicked.connect(resize_function)

        #槽函数
        def in_path():
            path = QFileDialog.getExistingDirectory(self,"选择一个文件夹",'./')
            lable3.setText("当前工作路径："+path)

        btn1.clicked.connect(in_path)

        # 槽函数，获取输入的像素值
        def get_pxy():
            pxy_str = lineEdit1.text()
            lable7.setText(pxy_str)

        btn2.clicked.connect(get_pxy)

        def get_pxx():
            pxx_str = lineEdit2.text()
            lable8.setText(pxx_str)

        btn3.clicked.connect(get_pxx)

        # 槽函数
        def out_path():
            path = QFileDialog.getExistingDirectory(self, "选择一个文件夹", './')
            lable11.setText("当前工作路径：" + path)

        btn4.clicked.connect(out_path)

        #3. 调用功能函数
        #in_path=in_path();
        #out_path=out_path();
        #pxy = get_pxy()
        #pxx = get_pxx()
        #setting= [pxx,pxy,in_path,out_path]
        #return setting


#2.测试代码
if __name__ == '__main__':

    #2.1 创建一个应用程序对象
    app = QApplication(sys.argv)

    #2.2 创建窗口
    window = Resize_Window()
    #2.3 展示窗口
    window.show()
    #2.4 应用程序的执行，进入到消息循环
    sys.exit(app.exec_())