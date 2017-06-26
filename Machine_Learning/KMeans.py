#! /usr/bin/python

import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
import numpy as np
from sklearn.cluster import KMeans

X = np.array([[1,2],[1.5,1.8],[5,8],[8,8],[1,0.6],[9,11]])

#plt.plot(X[:,0], X[:,1],'bo')
#plt.show()

clf = KMeans(n_clusters=2)
clf.fit(X)

centroids = clf.cluster_centers_
labels = clf.labels_

print(centroids)
print(labels)

colors = ["g.","r.","c.","b.","k."]
colors_center = ['g','r','c','b','k']

for i in range(len(X)):
	plt.plot(X[i][0], X[i][1], colors[labels[i]], markersize = 10)

k = 0	
for i in centroids:
	plt.scatter(i[0], i[1], c=colors_center[k], marker='x', s=150)
	k += 1

plt.show()
