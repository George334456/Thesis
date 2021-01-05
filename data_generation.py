from typing import Tuple, List
from random import uniform
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import heapq
import math
import pickle
import pdb

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

def run_k_means(lst, K):
    """
    Take a array of dimension points and run K-means clustering on it. Creates K clusters

    Returns a dictionary of K lists of points that are clustered according to K-means and the clusters.
    Note that the list of points at index i has center at centers[i]
    """
    dimension = lst.shape[1]
    kmeans = KMeans(n_clusters = K, random_state = 0).fit(lst)
    centers = kmeans.cluster_centers_
    
    # Using the centers, we assign each point to the closest center.
    result = {}
    for i in lst:
        index = kmeans.predict(i.reshape((1, dimension)))[0]

        if index in result:
            result[index].append(i)
        else:
            result[index] = [i]

    for res in result:
        result[res] = np.asarray(result[res])

    return result


def distance(p1, p2, dimension):
    """
    Returns the distance between two points.

    Dimensions is the number of dimensions that the two points are in.
    """
    ans = 0
    for i in range(dimension):
        ans += (p1[i] - p2[i]) ** 2
    return math.sqrt(ans)

def unpickle(file):
    """load the cifar-10 data"""

    with open(file, 'rb') as fo:
        data = pickle.load(fo, encoding='bytes')
    return data

def create_image_6d(file_name, output, need_pickle=False):
    pdb.set_trace()
    full_data = []

    first = [(x,y) for x in (4, 12) for y in (4,12)]
    second = [(x,y) for x in (20, 28) for y in (20, 28)]
    for i in range(1):
        if need_pickle:
            data = unpickle(f'cifar-100-python/train')[b'data']
        else:
            data = pickle.load(open(file_name, 'rb'), encoding='latin1')


        pdb.set_trace()
        # Data is an np array of N x 3072.
        print(data)

        for i in data:
            red = i[0:1024].reshape(32, 32)
            green = i[1024: 2048].reshape(32, 32)
            blue = i[2048:].reshape(32, 32)

            a,b,c,d = first
            full_data.append(tuple([i[a] for i in [red, green, blue]] + [i[b] for i in [red, green, blue]]))
            full_data.append(tuple([i[c] for i in [red, green, blue]] + [i[d] for i in [red, green, blue]]))
            print([i[a] for i in [red, green, blue]] + [i[b] for i in [red, green, blue]])

            
            a,b,c,d = second
            full_data.append(tuple([i[a] for i in [red, green, blue]] + [i[b] for i in [red, green, blue]]))
            full_data.append(tuple([i[c] for i in [red, green, blue]] + [i[d] for i in [red, green, blue]]))
            print([i[a] for i in [red, green, blue]] + [i[b] for i in [red, green, blue]])

        pdb.set_trace()
    full_data = list(full_data)

    full_data = np.asarray(full_data)
    pickle.dump(full_data, open(f'{output}', 'wb'))
    

    print(full_data.shape)



if __name__ == '__main__':
    create_image_6d('cifar10-ready.bin', '6d_LEGO')
    # a = generate_numbers(0, 255, 100, 6)
    # pickle.dump(a, open('qpoints_images.dump', 'wb'))

    
