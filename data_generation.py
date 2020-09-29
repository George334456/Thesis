from typing import Tuple, List
from random import uniform
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import heapq
import math

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
    lst = generate_numbers(50, 60, 3)
    lst2 = generate_numbers(0, 10, 10)
    lst = np.vstack((lst, lst2))
    X = np.take(lst, 0, 1) # TAke along the y axis.
    Y = np.take(lst, 1, 1)
    lst1, lst2, kmeans = cluster(lst, 10)
    
    # print(lst1)
    # print(lst2)
    print(kmeans.cluster_centers_)
    
    X_1 = np.take(lst1,0,1)
    Y_1 = np.take(lst1,1,1)

    X_2 = np.take(lst2, 0, 1)
    Y_2 = np.take(lst2, 1, 1)
    
    plt.plot(X, Y, '.', color='black')

    plt.plot(X_1 , Y_1, '.', color='green')
    plt.plot(X_2, Y_2, '.', color='blue')
    cluster_x = np.take(kmeans.cluster_centers_, 0, 1)
    cluster_y = np.take(kmeans.cluster_centers_, 1, 1)
    plt.plot(cluster_x, cluster_y, '.', color='red')
    plt.show()
