from typing import Tuple, List
from random import uniform
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import heapq
import math
import pickle

def rebalance(dest, src, dest_center, src_center, m, dimension):
    """
    Take elements from src to dest based on how many destination is missing from M//2.
    Inserts until dest has >= m elements in it.

    Takes dimensions as an input

    Returns a tuple of lists representing the two lists after movement of points.
    """
    distances = []
    to_delete = []
    for i,e in enumerate(src):
        heapq.heappush(distances, (distance(dest_center, e, dimension), i))
    while len(dest) < m:
        smallest = heapq.heappop(distances)
        ind = smallest[1]
        dest = np.vstack((dest, src[ind]))
        to_delete.append(ind)
    print(to_delete)
    src = np.delete(src, to_delete, 0)
    return (dest, src)

def generate_numbers(minimum: int, maximum: int, amount: int, dimension: int):
    """
    Randomly generate amount of tuples ranging from minimum to maximum.
    Creates points of dimension long.

    Returns a (amount x dimension) long array.
    """
    lst = np.zeros((amount, dimension))
    for i in range(amount):
        for j in range(dimension):
            lst[i][j] = uniform(minimum, maximum)
    return lst

def cluster(lst, M):
    """
    Take a array of dimension points and run K-means clustering on it.
    M is the minimum number assigned to a cluster
    Returns 2 lists of points that are clustered according to K-means and the k-means object
    """
    dimension = lst.shape[1]
    kmeans = KMeans(n_clusters = 2, random_state = 0).fit(lst) # TODO: Do I keep random_state in for determinism?
    centers = kmeans.cluster_centers_

    lst1 = []
    lst2 = []
    for i in lst:
        if kmeans.predict(i.reshape([1, dimension])) == 0:
            lst1.append(i)
        else:
            lst2.append(i)

    lst1 = np.asarray(lst1)
    lst2 = np.asarray(lst2)

    if lst1.shape[0] < M//2:
        lst1, lst2 = rebalance(lst1, lst2, centers[0], centers[1], M//2, dimension)
    elif lst2.shape[0] < M//2:
        lst2, lst1 = rebalance(lst2, lst1, centers[1], centers[0], M//2, dimension)
    return (lst1, lst2, kmeans)

def distance(p1, p2, dimension):
    """
    Returns the distance between two points.

    Dimensions is the number of dimensions that the two points are in.
    """
    ans = 0
    for i in range(dimension):
        ans += (p1[i] - p2[i]) ** 2
    return math.sqrt(ans)


if __name__ == '__main__':
    for dimension in [3, 4, 5, 6]:
        for i in [1000, 4000, 8000, 16000, 32000, 64000]:
            lst = generate_numbers(0, 8000, i, dimension)
            pickle.dump(lst, open(f"synthetic_{i}_{dimension}d.dump", "wb"))

    
