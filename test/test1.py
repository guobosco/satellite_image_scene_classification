import cv2,os
import numpy as np
import time

classes = ["agricultural", "airplane", "baseballdiamond",
                       "beach", "buildings", "chaparral", "denseresidential",
                       "forest", "freeway", "golfcourse", "harbor", "intersection",
                       "mediumresidential", "mobilehomepark", "overpass", "parkinglot",
                       "river", "runway", "sparseresidential", "storagetanks", "tenniscourt"]
read_path = "/Users/mac/Documents/GitHub/satellite_image_scene_classification/classification/Images"
# 写入路径
write_path ="/Users/mac/Desktop/12"
for fields in classes:
    path = os.path.join(read_path, fields)
    w_path = os.path.join(write_path, fields)
    file_names = os.listdir(path)
    for file_name in file_names:
        img = cv2.imread(os.path.join(path, file_name) , 1)
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
        #cv2.imwrite('/Users/mac/Desktop/Jan_23/Images_uniformization', img)
        ##print(os.path.join(w_path, file_name))

        #print("当前时间为：" + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))