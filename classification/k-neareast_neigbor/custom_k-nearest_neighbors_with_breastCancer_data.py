import numpy as np
from math import sqrt
import warnings
from collections import Counter
import pandas as pd
import random

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
    confidence = Counter(votes).most_common(1)[0][1] / k
    #print(vote_result, confidence)

    #another form of it
    #for i in sorted(distances)[:k]:
    #    i[i]

    return vote_result, confidence

accuracies=[]
for i in range(25) : # compare the accuracies
    df = pd.read_csv('breast-cancer-wisconsin.data')
    df.replace('?', -99999, inplace=True)
    df.drop(['id'], 1, inplace=True)
    #print(df.head())
    #all the data needs to be turned to a float because it has some strange number presented as a '' format
    full_data = df.astype(float).values.tolist()
    #print(full_data[:5])
    random.shuffle(full_data)
    #print(20*'*')
    #print(full_data[:5])

    test_size = 0.4
    train_set = {2:[], 4:[]}
    test_set = {2:[], 4:[]}
    #train data will be the first 20% of the full_data
    train_data = full_data[:-int(test_size * len(full_data))]

    #test data will be the last 20% of the full_data
    test_data = full_data[-int(test_size * len(full_data)):]

    #popoulate data to the dictionaries
    for i in train_data:
        # -1 is the last element or last value of the full_data
        # we are appending list, to the this list element is upto the last element.
        train_set[i[-1]].append(i[:-1])

    for i in test_data:
        # -1 is the last element or last value of the full_data
        # we are appending list, to the this list element is upto the last element.
        test_set[i[-1]].append(i[:-1])

    correct = 0
    total = 0
    for group in test_set:
        for data in test_set[group]:
            vote, confidence = k_nearest_neighbors(train_set, data, k=5)
            if group == vote:
                correct +=1
            else:
                # test group and train group are not matched so they need to be looked at in details.
                print('Confidence:', confidence)
            total +=1

    print('Accuracy:', correct/total)
    accuracies.append(correct/total)

print(sum(accuracies)/len(accuracies))