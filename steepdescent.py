import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas import Series, DataFrame
from numpy.random import normal
from sklearn.datasets import load_boston
from mpl_toolkits.mplot3d import axes3d, Axes3D

boston = load_boston()
boston_target = boston.target 
boston_data = boston.data

boston_CRIM = []

for i in range(boston_data.shape[0]):
    boston_CRIM.append(boston_data[i][0])

def create_dataset(data, target, num):
    dataset = DataFrame(columns=['x', 'y'])
    for i in range(num):
        x = data[i]
        y = target[i]
        dataset = dataset.append(Series([x, y], index=['x','y']),
                                ignore_index=True)
    return dataset

boston_dataset = create_dataset(boston_CRIM, boston_target, 168)

print(boston_dataset.shape)
print(boston_dataset)


def calSteep(w0, w1):
    result = []
    r0 = 0
    r1 = 0

    for i in range(boston_dataset.shape[0]):
        x = boston_dataset.x[i]
        t = boston_dataset.y[i]
        r0 += w0 + w1 * x - t
        r1 += (w0 + w1 * x - t) * x

    result = [r0, r1]
    return result

def ed_error(dataset, f):
    err = 0.0
    for index, line in dataset.iterrows():
        x, y = line.x, line.y
        # print('x is : ' + str(x))
        # print('y is : '  + str(y))
        err += 0.5 * (f(x) - y)**2
    return err
    
origin_num_0 = 1
origin_num_1 = 3

sequence_origin0 = []
sequence_origin1 = []
sequence_ed = []

for i in range(200):
    steepPair = []

    def f(x):
        res = origin_num_0 + origin_num_1 * x
        return res
    
    ed = ed_error(boston_dataset, f)
    sequence_ed.append(ed)

    steepPair = calSteep(origin_num_0, origin_num_1)
    origin_num_0 = origin_num_0 - 0.001 * steepPair[0]
    origin_num_1 = origin_num_1 - 0.001 * steepPair[1]

    sequence_origin0.append(origin_num_0)
    sequence_origin1.append(origin_num_1)

plt.plot(sequence_origin0)
plt.show()

plt.plot(sequence_origin1)
plt.show()

mesh_sequence_origin0, mesh_sequence_origin1 = np.meshgrid(sequence_origin0, sequence_origin1)
np_sequence_ed = np.array(sequence_ed)
sequence_origin0, np_sequence_ed = np.meshgrid(sequence_origin0, np_sequence_ed)

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
surf = ax.plot_surface(mesh_sequence_origin0, mesh_sequence_origin1, np_sequence_ed,
                       linewidth=0, antialiased=False)

plt.show()
