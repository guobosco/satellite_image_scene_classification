import os,cv2
import numpy as np
from PyQt5 import QtCore
import sys
#0. 导入包和模块
from PyQt5.Qt import *

import resize_main,uniformization



if __name__ == '__main__':


    #2.1 创建一个应用程序对象
    app = QApplication(sys.argv)

    #2.2 创建窗口
    window1 = resize_main.Resize_Window()
    window2 = uniformization.Uniformization_Window()

    #2.3 展示窗口
    window1.show()
    window2.show()
    window1.move(200,200)
    window2.move(750,200)

    #2.4 应用程序的执行，进入到消息循环
    sys.exit(app.exec_())