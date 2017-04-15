import numpy as np
from math import sqrt
import matplotlib.pyplot as plt
import warnings
from matplotlib import style
from collections import Counter

style.use('fivethirtyeight')

#features
dataset = {'k':[[1,2],[2,3],[3,1]], 'r':[[6,5],[7,7],[8,6]]}
new_features = [5,7]


def k_nearest_neighbors(data, predict, k=3):
    if len(data) >= k:
        warnings.warn('K is set to a value less than total votng groups')
    distances = []
    for group in data:
        for features in data[group]:
            # euclidean distance by using np array //dynamic and faster
            # euclidean_distance = np.sqrt(np.sum((np.array(features) - np.array(predict))**2))
            # euclidean distance by using short form of in np //dynamic and faster
            euclidean_distance = np.linalg.norm(np.array(features) - np.array(predict))
            # print(group)
            # print(features)
            # print(predict)
            # print(euclidean_distance)
            distances.append([euclidean_distance, group])

    votes = [i[1] for i in sorted(distances)[:k]] #sort the distances 0 upto value of K

    '''List the n most common elements and their counts from the most
        common to the least.  If n is None, then list all element counts.

        >>> Counter('abcdeabcdabcaba').most_common(3)
        [('a', 5), ('b', 4), ('c', 3)]

    It is a TUPLE '''
    vote_result = Counter(votes).most_common(1)[0][0]


    #another form of it
    #for i in sorted(distances)[:k]:
    #    i[i]

    return vote_result

result = k_nearest_neighbors(dataset, new_features, k=3)
print(result)


## another implementation of below for loop is :
#[[plt.scatter(i_i[0], i_i[1], s=100, color= i) for i_i in dataset[i]] for i in dataset]

for i in dataset:
    for i_i in dataset[i]:
        plt.scatter(i_i[0], i_i[1], s=100, color= i) # s = size
plt.scatter(new_features[0], new_features[1], s=100, color= result)
plt.show()