#! /usr/bin/python

import numpy as np
from math import sqrt
import matplotlib.pyplot as plt
import warnings
from matplotlib import style
from collections import Counter

style.use('fivethirtyeight')

dataset = {'k': [[1,2],[2,3],[3,1]], 'r':[[6,5],[7,7],[8,6]]}
new_features = [5,7]

#for i in dataset:
#	for ii in dataset[i]:
#		plt.scatter(ii[0],ii[1],s=100,color=i)

#[[plt.scatter(ii[0],ii[1], s=100, color=i) for ii in dataset[i]] for i in dataset]
#plt.scatter(new_features[0],new_features[1], s=100)
#plt.show()

def k_nearest_neighbors(data, predict, k=3):
	if len(data) >=k:
		warnings.warn('K is set to a value less than total voting groups!')
	distances = []
	for group in data:
		for features in data[group]:
			#euclidean_distance = sqrt((features[0]-predict[0])**2+(features[1]-predict[1])**2)
			#euclidean_distance = np.sqrt(np.sum((np.array(features)-np.array(predict))**2))
			eucleadian_distance = np.linalg.norm(np.array(features)-np.array(predict))
			distances.append([eucleadian_distance, group])
			print(distances)

	votes = [i[1] for i in sorted(distances)[:k]]
	#print(votes)
	print(Counter(votes).most_common(1))
	vote_results = Counter(votes).most_common(1)[0][0]

	return vote_results

result = k_nearest_neighbors(dataset, new_features, k=3)
print(result)

		

