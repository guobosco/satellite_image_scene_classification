#0. 导入包和模块
from PyQt5.Qt import *
import sys
import cv2
import resize_256to224
import os

#1. 创建窗口类
class Window(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("改变输入图片的大小")
        self.resize(500,500)
        self.setup_ui()

    #设置窗口ui
    def setup_ui(self):
        #1. 创建控件,例如：lable = QLabel(self)
        lable1 = QLabel(self)

        #2. 设置控件
        lable1.move(50,50)
        lable1.setText("设置输入路径：")
        #3. 调用功能函数

    #写功能函数
    def QObject_test(self):
        pass

"""修改图片大小的主函数"""
def resize_function(read_path,write_path):
    #源文件路径
    #'/Users/mac/Documents/GitHub/satellite_image_scene_classification/UCMerced_LandUse'
    #'/Users/mac/Documents/GitHub/satellite_image_scene_classification/UCMerced_LandUse(224)'
    read_path = read_path
    #写入路径
    write_path =  write_path
    #得到文件名字符串列表img_list_name
    img_list =  os.listdir(read_path)
    img_list_name = []
    for i in range(0,len(img_list)):
        img_list_name.append(os.path.basename(img_list[i]))

    #逐个对每个图像文件处理大小resize
    for img_name in img_list_name:
        if img_name[-4:-1] == '.tif':       #确保是以.tif结尾的图像文件
            img256 = cv2.imread(read_path + "/" + img_name)
            img224 = cv2.resize(img256, (224, 224))         #本程序核心代码
            cv2.imwrite(write_path+ "/" + img_name,img224)


#2.测试代码
if __name__ == '__main__':

    #2.1 创建一个应用程序对象
    app = QApplication(sys.argv)

    #2.2 创建窗口
    window = Window()
    #2.3 展示窗口
    window.show()


    #2.4 应用程序的执行，进入到消息循环
    sys.exit(app.exec_())