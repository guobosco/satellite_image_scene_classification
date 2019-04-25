import cv2

gray = cv2.imread('/Users/mac/Desktop/hitlight.png',0)
cv2.imshow("original",gray)


cv2.waitKey()
cv2.destroyAllWindows()
