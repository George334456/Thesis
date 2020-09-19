from typing import Tuple, List
from random import uniform
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt

def generate_numbers(minimum: int, maximum: int, amount: int):
    # Randomly generate "amount" amount of tuples ranging from minimum to maximum
    # Returns a (amount x 2) np array
    lst = np.zeros((amount, 2))
    for i in range(amount):
        lst[i][0] = uniform(minimum, maximum) # Should be x coordinate of i-th data
        lst[i][1] = uniform(minimum, maximum) # Should be y coordinate of i-th data
    return lst

def cluster(lst):
    # Take a 2d array and run k-means clustering on it.
    # Returns the clustered centers
    kmeans = KMeans(n_clusters = 2, random_state = 0).fit(lst) # TODO: Do I keep random_state in for determinism?
    return kmeans.cluster_centers_


if __name__ == '__main__':
    lst = generate_numbers(0, 100, 100)
    print(lst)
    X = np.take(lst, 0, 1) # TAke along the y axis.
    Y = np.take(lst, 1, 1)
    print(cluster(lst))
    plt.plot(X, Y, '.', color='black')
    cluster_x = np.take(cluster(lst), 0, 1)
    cluster_y = np.take(cluster(lst), 1, 1)
    plt.plot(cluster_x, cluster_y, '.', color='red')
    plt.show()
