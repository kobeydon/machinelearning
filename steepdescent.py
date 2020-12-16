import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas import Series, DataFrame
from numpy.random import normal
from sklearn.datasets import load_boston


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

boston_dataset = create_dataset(boston_CRIM, boston_target, 506)

print(boston_dataset.shape)
print(boston_dataset.x)

plt.scatter(boston_dataset.x, boston_dataset.y, marker='o', color='blue')

plt.show()