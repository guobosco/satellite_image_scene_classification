import cv2
import os
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

