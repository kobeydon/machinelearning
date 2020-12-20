import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas import Series, DataFrame
from numpy.random import normal

pos = [0, 0.111111, 0.222222, 0.333333, 0.444444, 0.555556, 0.666667, 0.777778, 0.888889, 1]
value = [0.659981, 0.849493, 1.853337, 1.500273, 0.297486, 0.10194, -0.2152, -0.66527, 0.107402, 0.098598]

def createdataset(xl, yl):
    dataset = DataFrame(columns=['x', 'y'])
    for i in range(10):

        x = xl[i]
        y = yl[i]

        dataset = dataset.append(Series([x, y], index=['x','y']),
                                    ignore_index=True)
    return dataset

def calSteep(w0, w1, dataset):
    result = []
    r0 = 0
    r1 = 0

    for i in range(10):
        x = dataset.x[i]
        t = dataset.y[i]
        r0 += w0 + w1 * x - t
        r1 += (w0 + w1 * x - t) * x

    result = [r0, r1]
    return result

def f(x):
    res = 0.0
    res = 1 + 3 * x
    return res

def ed_error(dataset, func):
        err = 0.0
        for index, line in dataset.iterrows():
            x, y = line.x, line.y
            # print('x is : ' + str(x))
            # print('y is : '  + str(y))
            err += 0.5 * (func(x) - y)**2
        return err

plsdog_dataset = createdataset(pos, value)

edres = ed_error(plsdog_dataset, f)

print(edres)