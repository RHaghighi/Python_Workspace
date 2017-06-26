#! /usr/bin/python

import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from sklearn.datasets.samples_generator import make_blobs
from sklearn.cluster import MeanShift


centers = [[1,1,1],[5,5,5],[3,10,10]]
X, _ = make_blobs(n_samples=500, centers=centers, cluster_std=1)

ms = MeanShift()
ms.fit(X)

centroids = ms.cluster_centers_
labels = ms.labels_

print(centroids)
print(labels)

colors = 10*["g","r","c","b","k"]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

for i in range(len(X)):
        ax.scatter(X[i][0], X[i][1], X[i][2], c=colors[labels[i]], marker='o')

k = 0
for i in centroids:
        ax.scatter(i[0], i[1], i[2], c=colors[k], marker='x', s=150)
        k += 1

plt.show()

