import cv2
o=cv2.imread("/Users/mac/Desktop/interim_report/test_img/test_img_aircraft.jpg")
r=cv2.GaussianBlur(o,(3,3),0)
cv2.imshow("original",o)
cv2.imshow("result",r)
cv2.waitKey()
cv2.destroyAllWindows()