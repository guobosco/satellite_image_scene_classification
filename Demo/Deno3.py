import cv2
import numpy as np
o = cv2.imread('/Users/mac/Desktop/WechatIMG76.jpeg',cv2.IMREAD_GRAYSCALE)
sobelx = cv2.Sobel(o,cv2.CV_64F,1,0,ksize=3)
sobely = cv2.Sobel(o,cv2.CV_64F,0,1,ksize=3)
sobelx = cv2.convertScaleAbs(sobelx)   # 转回uint8
sobely = cv2.convertScaleAbs(sobely)
sobelxy =  cv2.addWeighted(sobelx,0.5,sobely,0.5,0)
cv2.imshow("original",o)
cv2.imshow("xy",sobelxy)
cv2.waitKey()
cv2.destroyAllWindows()
