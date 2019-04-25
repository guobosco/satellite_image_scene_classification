import numpy as np
from matplotlib import pyplot as plt

x=np.linspace(-10,10,200)
y=((np.exp(x))-(np.exp(-x)))/((np.exp(x))+(np.exp(-x)))
plt.figure(1)

ax = plt.gca()
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')

ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
ax.spines['bottom'].set_position(('data', 0))   #指定 data  设置的bottom(也就是指定的x轴)绑定到y轴的0这个点上
ax.spines['left'].set_position(('data', 0))

plt.plot(x,y,linewidth='1.5',color='black')
plt.savefig("/Users/mac/Desktop/figure/filename2.png")

plt.show()