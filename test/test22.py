from PIL import Image
from keras.preprocessing import image
import glob
import numpy as np
from collections import Counter

IMG_W, IMG_H, IMG_CH = 150, 150, 3  # 单张图片的大小
image_path = "/Users/mac/Desktop/cut"
sigle_image_path = glob.glob(image_path + '/*.jpg', recursive=True)
res_label = ["飞机", "机场", "棒球场",
                 "篮球场", "沙滩", "桥梁", "丛林",
                 "教堂", "圆形农场", "云朵", "商业区", "密集居民区",
                 "沙漠", "森林", "高速公路", "高尔夫球场","田径场",
                 "码头", "工业区", "十字路口", "岛屿", "湖泊",
                 "草地", "中等住宅", "房车停车场", "方形农田", "立交桥", "宫殿", "停车场",
                 "高速公路", '火车站', "方形农田", "中等住宅", "环形转盘", "跑道", "海上冰山", "舰船",
                 "冰山", "独立房屋", "体育场", "储罐", "网球场", "梯田", "热电站", "湿地"]

def predict(model, img_path, target_size):
    img = Image.open(img_path)  # 加载图片
    if img.size != target_size:
        img = img.resize(target_size)

    x = image.img_to_array(img)
    x *= 1. / 255  # 相当于ImageDataGenerator(rescale=1. / 255)
    x = np.expand_dims(x, axis=0)  # 调整图片维度
    preds = model.predict(x)  # 预测
    return preds[0]

from keras.models import load_model
saved_model = load_model('/Users/mac/Documents/GitHub/models/summary_module/NWPU/self_design_NWPU45.h5')
res = []
for imag in sigle_image_path:
    result = predict(saved_model, imag, (IMG_W, IMG_H))
    lable_res=res_label[result.argmax()]
    res.append(lable_res)
    print(lable_res)
res_f = Counter(res)
res_f = res_f.most_common()
print(res_f)
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False
count = [res_f[0][1],res_f[1][1],res_f[2][1],res_f[3][1],res_f[4][1],res_f[5][1],res_f[6][1],res_f[7][1]]
plt.barh(range(8), count, height=0.7, color='blue', alpha=0.8)
plt.yticks(range(8), [res_f[0][0], res_f[1][0], res_f[2][0], res_f[3][0], res_f[4][0],res_f[5][0],res_f[6][0],res_f[7][0]])
plt.xlim(0,200)
plt.xlabel("次数")
plt.title("分割图片类别统计")
for x, y in enumerate(count):
    plt.text(y + 0.2, x - 0.1, '%s' % y)
plt.show()
plt.savefig("/Users/mac/Desktop/zhifangtu/fake.png")