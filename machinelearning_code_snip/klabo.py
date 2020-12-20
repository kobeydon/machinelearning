# coding : utf-8

from sklearn.datasets import load_boston

datasets = load_boston()

data = datasets.data
targets = datasets.target
feature_name = datasets.feature_names

print(feature_name)
print(targets)

print("====配列====")
print(data[2][0])


