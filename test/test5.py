import pandas as pd
import numpy as np
import requests
import seaborn as sns

r = requests.get('http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data')

with open('iris.data','w') as f:
    f.write(r.text)

import pandas as pd
data = pd.read_csv('iris.data',names = ['e_cd','e_kd','b_cd','b_kd','cat'])
data['c1'] = np.array(data['cat'] == 'Iris-setosa').astype(np.float32)
data['c2'] = np.array(data['cat'] == 'Iris-versicolor').astype(np.float32)
data['c3'] = np.array(data['cat'] == 'Iris-virginica').astype(np.float32)

target =  np.stack([data.c1.values,data.c2.values,data.c3.values]).T
shuju = np.stack([data.e_cd.values,data.e_kd.values,data.b_cd.values,data.b_kd.values]).T


