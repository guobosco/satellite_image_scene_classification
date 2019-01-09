import cv2

img256 = cv2.imread('/Users/mac/Documents/GitHub'
                 '/satellite_image_scene_classification/img_preprocessing'
                 '/test_imgs/agricultural00.tif')

img224 = cv2.resize(img256,(224,224))

cv2.imwrite('/Users/mac/Documents'
            '/GitHub/satellite_image_scene_classification/img_preprocessing'
            '/test_imgs/agricultural00.png',img224)