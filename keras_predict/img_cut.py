"""
将一张图片均匀向下移动，向右移动，每移动一次，切割出一个固定面积的图片
算法思想：我们知道图片实际上是有一个二维数组组成的，所以先控制横坐标不变，纵坐标截取，一直到纵坐标的边界，然后向右移动横坐标，重复上一步操作
"""
# python实现图片切割技术
from PIL import Image

# id标识图片 vx,vy为切成的图片大小
def cut(CUT_PATH, BLOCK_W, BLOCK_H,BLOAK_PATH):
    cut_image = CUT_PATH
    print(cut_image)
    cut_image_name = cut_image.split('/')[-1].split('.')[0]
    block_images = BLOAK_PATH + "/" + cut_image_name +"_block_"
    im = Image.open(cut_image)

    dx = 40                                         # 偏移量
    dy = 40
    n = 1                                           # 切割图片的片数
    x1 = 0                                          # 起点x
    y1 = 0                                          # 起点y
    x2 = BLOCK_W                                    # 终点x
    y2 = BLOCK_H                                    # 终点y
    print(im.size)                                  # 输出图片的大小    
    cut_w = im.size[0]                              # 将原图的宽赋值给w
    cut_h = im.size[1]                              # 将原图的高赋值给h
    while x2 <= cut_w:
        # 先纵向切割
        while y2 <= cut_h:
            block_image = block_images + str(n) + ".jpg"
            # crop() : 从图像中提取出某个矩形大小的图像。它接收一个四元素的元组作为参数，各元素为（left, upper, right, lower），坐标系统的原点（0, 0）是左上角。
            im2 = im.crop((x1, y1, x2, y2))
            im2.save(block_image)
            y1 = y1 + dy
            y2 = y1 + BLOCK_H
            n = n + 1                               # 更新n的值
        x1 = x1 + dx
        x2 = x1 + BLOCK_W
        y1 = 0
        y2 = BLOCK_H
    print("图片切割成功，切割得到的子图片数为：")
    return n - 1

if __name__ == "__main__":
    CUT_PATH = "/Users/mac/Desktop/PIL_TEST.jpeg"   # 图片存储绝对路径
    BLOCK_W = 224
    BLOCK_H = 224
    BLOAK_PATH = "/Users/mac/Desktop/cut"
    res = cut(CUT_PATH, BLOCK_W, BLOCK_H,BLOAK_PATH)
    print(res)