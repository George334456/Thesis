from typing import Tuple, List
from random import uniform
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import heapq
import math
import pickle

def rebalance(dest, src, dest_center, src_center, m):
    """
    Take elements from src to dest based on how many destination is missing from M//2.
    """
    distances = []
    to_delete = []
    for i,e in enumerate(src):
        heapq.heappush(distances, (distance(dest_center, e), i))
    while len(dest) < m:
        smallest = heapq.heappop(distances)
        ind = smallest[1]
        dest = np.vstack((dest, src[ind]))
        to_delete.append(ind)
    print(to_delete)
    src = np.delete(src, to_delete, 0)
    return (dest, src)


def generate_numbers(minimum: int, maximum: int, amount: int):
    # Randomly generate "amount" amount of tuples ranging from minimum to maximum
    # Returns a (amount x 2) np array
    lst = np.zeros((amount, 2))
    for i in range(amount):
        lst[i][0] = uniform(minimum, maximum) # Should be x coordinate of i-th data
        lst[i][1] = uniform(minimum, maximum) # Should be y coordinate of i-th data
    return lst

def cluster(lst, M):
    # Take a 2d array and run k-means clustering on it.
    # M is the minimum number assigned to a cluster
    # Returns 2 lists of points that are clustered according to K-means, and the k-means object
    kmeans = KMeans(n_clusters = 2, random_state = 0).fit(lst) # TODO: Do I keep random_state in for determinism?
    centers = kmeans.cluster_centers_

    lst1 = []
    lst2 = []
    for i in lst:
        if kmeans.predict(i.reshape([1,2])) == 0:
            lst1.append(i)
        else:
            lst2.append(i)

    lst1 = np.array(lst1)
    lst2 = np.array(lst2)

    if lst1.shape[0] < M//2:
        lst1, lst2 = rebalance(lst1, lst2, centers[0], centers[1], M//2)
    elif lst2.shape[0] < M//2:
        lst2, lst1 = rebalance(lst2, lst1, centers[1], centers[0], M//2)
    return (lst1, lst2, kmeans)

def distance(p1, p2):
    return math.sqrt( ((p1[0] - p2[0]) ** 2) + ((p1[1] - p2[1]) ** 2))


if __name__ == '__main__':
    for i in [1000, 4000, 8000, 16000, 32000, 64000]:
        lst = generate_numbers(0, 8000, i)
        pickle.dump(lst, open(f"synthetic_{i}.dump", "wb"))

    
