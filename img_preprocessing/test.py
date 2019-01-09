import cv2
import os

#源文件路径
read_path = '/Users/mac/Documents/GitHub' \
            '/satellite_image_scene_classification/UCMerced_LandUse'
img_list =  os.listdir(read_path)
img_list_name = []
for i in range(0,len(img_list)):
    img_list_name.append(os.path.basename(img_list[i]))
