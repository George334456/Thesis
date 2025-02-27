import numpy as np
import math
from data_generation import generate_numbers, run_k_means
from shard_implementation import Shard, Shard_index, Cell
from rtree import min_max_dist, min_dist, prune, bounding_rectangle
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from functools import reduce, cmp_to_key
import kd_tree as kd
import pdb
import pickle
import time
import operator
import cupy as cp
import sys, getopt
"""
Quick Reminders:
Theta = Border points generated along every axis
T_i = length of the border points along each axis
lst = list of points to be trained on.
k = Number of partitions for the mapped values. Essentially equivalent to how many F we will have.
sigma = Set of breakpoints for beta
V = Number of points in a particular partition.
psi = estimated number of keys in a shard. Approximately equal to how many keys can fit into a page.
M = the partition list of mapped values that is used to train each individual F_i
"""
def check_alpha(alpha):
    """
    Checks if alpha satisfies the fact that we should be continuously increasing.
    """
    alpha = alpha.flatten()[1:]
    for i in range(len(alpha)):
        if cp.sum(alpha[:i+1]) < 0:
            return False
    return True

def calculate_loss(A, alpha, y):
    """
    Calculate the loss function from A, alpha, y

    A, alpha, and y are cupy arrays
    """
    return cp.sum(cp.power(A @ alpha - y, 2))

def dfs(list_of_list, max_depth, depth):
    """
    Grabs tuples from list_of_list starting at depth and going until the last depth.

    Return all possible combinations of the tuples along each dimension axis.

    This is a depth first traversal of list of lists.

    list_of_list is a list of lists. len(list_of_list) == dimension. Each individual list can contain variable length, but the list will always be a list of tuples.
    We want to grab all possible tuples that take one tuple from each dimension
    max_depth is the maximum depth of the list of lists. Note that max_depth == dimension - 1
    depth is the current depth of the list of list visit.
    """
    full_tuples = []
    if depth == max_depth:
        for i in list_of_list[depth]:
            full_tuples.append([i])
        return full_tuples
    for i in list_of_list[depth]:
        combinations = dfs(list_of_list, max_depth, depth + 1)
        for j in combinations:
            j.insert(0, i)
            full_tuples.append(j)
    return full_tuples

def old_train(partitions, sigma, psi):
    # Change psi as we increase the number of points it can take.
    settings = []

    # TODO: Make it so that we have more than 5 points in a partition. See if beta can descend when we have more to work with.

    for partition in partitions:
        V = len(partition)
        y = cp.arange(V).reshape([V, 1])
        beta = cp.zeros([1,sigma])
        for i in range(sigma):
            ind = cp.floor(i * V/sigma)
            # ind = np.floor(i * V/psi)
            beta[0][i] = partition[int(ind)] # Initialize with an integer

        try:
            alpha, A = calculate_alpha(partition, beta, y)
        except:
            pdb.set_trace()
            alpha, A = calculate_alpha(partition, beta,y )
            pass
        curr_loss = calculate_loss(A, alpha, y)
        orig_loss = curr_loss
        # print("loss")
        # print(loss)

        num_iterations = 0
        curr_beta_lr = None
        converge = False
        lr_lst = [0.1, 0.05, 0.01, 0.001]
        while(num_iterations < 1000):
            print(curr_loss)
            print(beta)
            meme = time.time()
            grad = calculate_gradient(alpha, beta, partition, A, y)
            print(f'{time.time() - meme} s for calculate gradient')
            print(num_iterations)
            print(grad)
            grad = grad.flatten()[1:] # Grab everything except the first element.
            meme = time.time()
            for i in lr_lst:
                print(f'Learning rate is {i}')
                beta_lr = (grad * i) + beta # Descend in the gradient. TODO: MAKE SURE THIS IS DOING ELEMENT WISE ADDITION
                beta_lr = cp.sort(beta_lr)
                try:
                    alpha, A = calculate_alpha(partition, beta_lr, y)
                except:
                    # If we fail to calculate_alpha properly, then we just say, okay, look at the next one.
                    continue
                if check_alpha(alpha):
                    loss = calculate_loss(A, alpha, y)
                    if loss < curr_loss:
                        curr_loss = loss
                        curr_beta_lr = beta_lr
                        # Since we're counting from max to low, we want to be greedy and break out immediately
                        break
            print(f'{time.time() - meme} for descent')
            if curr_beta_lr is not None:
                # We found something that had loss minimizing.
                beta = curr_beta_lr
                num_iterations += 1 
                if orig_loss - curr_loss < 0.001:
                    break
                orig_loss = curr_loss
            else:
                # Loss didn't change, so break?
                break
        alpha, A = calculate_alpha(partition, beta, y)
        settings.append((alpha.flatten(), beta.flatten(), (A @ alpha).flatten()))
    return settings

def train(partitions, sigma, psi):
    """
    Return the parameters for the piecewise monotone linear regression model.

    Partitions are a K x V array. Note that V might not be the same across all K. IE V is partition specific.
    sigma is the number of breakpoints we want. Is an int. In accordance to the paper, we pass it in as sigma + 1.
    psi is an integer representing the average number of things to 
    """
    # Change psi as we increase the number of points it can take.
    settings = []

    # TODO: Make it so that we have more than 5 points in a partition. See if beta can descend when we have more to work with.

    for partition in partitions:
        V = len(partition)
        y = cp.arange(V).reshape([V, 1])
        beta = cp.zeros([1,sigma])
        for i in range(sigma):
            ind = cp.floor(i * V/sigma)
            # ind = np.floor(i * V/psi)
            beta[0][i] = partition[int(ind)] # Initialize with an integer

        try:
            alpha, A = calculate_alpha(partition, beta, y)
        except:
            pdb.set_trace()
            alpha, A = calculate_alpha(partition, beta,y )
            pass
        curr_loss = calculate_loss(A, alpha, y)
        orig_loss = curr_loss
        # print("loss")
        # print(loss)

        num_iterations = 0
        curr_beta_lr = None
        converge = False
        lr_lst = [0.1, 0.05, 0.01, 0.001]
        while(num_iterations < 1000):
            print(curr_loss)
            print(beta)
            meme = time.time()
            grad = calculate_gradient(alpha, beta, partition, A, y)
            print(f'{time.time() - meme} s for calculate gradient')
            print(num_iterations)
            print(grad)
            grad = grad.flatten()[1:] # Grab everything except the first element.
            meme = time.time()
            while lr_lst:
                i = lr_lst[0]
                print(f'Learning rate is {i}')
                beta_lr = (grad * i) + beta # Descend in the gradient. TODO: MAKE SURE THIS IS DOING ELEMENT WISE ADDITION
                beta_lr = cp.sort(beta_lr)
                try:
                    alpha, A = calculate_alpha(partition, beta_lr, y)
                except:
                    # If we fail to calculate_alpha properly, then we just say, okay, look at the next one.
                    continue
                if check_alpha(alpha):
                    loss = calculate_loss(A, alpha, y)
                    if loss < curr_loss:
                        curr_loss = loss
                        curr_beta_lr = beta_lr
                        # Since we're counting from max to low, we want to be greedy and break out immediately
                        break
                lr_lst.pop(0)
            print(f'{time.time() - meme} for descent')
            if curr_beta_lr is not None:
                # We found something that had loss minimizing.
                beta = curr_beta_lr
                num_iterations += 1 
                if orig_loss - curr_loss < 0.001:
                    break
                orig_loss = curr_loss
            else:
                # Loss didn't change, so break?
                break
        alpha, A = calculate_alpha(partition, beta, y)
        settings.append((alpha.flatten(), beta.flatten(), (A @ alpha).flatten()))
    return settings

def fill_grid(partition, constant, comparison):
    partition = partition.flatten()
    row = np.zeros(len(partition))
    for i in range(len(partition)):
        row[i] = comparison(partition[i], constant)
    return row

def fill_A_row(partition, beta, comparison):
    """
    Fill a row in the A matrix.

    partition is an integer.
    Beta is a 1 x M array
    Comparison is a function to use for creating the row.
    """
    print(partition, beta, comparison)
    pdb.set_trace()
    row = np.zeros(beta.shape[1] + 1)
    row[0] = 0
    row[1:] = np.asarray([partition - beta if partition > beta else 0])
    return row


def calculate_gradient(alpha, beta, partition, A, y):
    """
    Calculates the gradient based on alpha/beta/partition/A/y

    alpha: A (1 x sigma + 2) array. Consists of a bar, a0, a1, a2, ..., a sigma. Therefore sigma + 2
    beta: A (1 x sigma + 1) array. Consists of b0, b1, ..., b_sigma. Therefore sigma + 1
    partition: A V length array representing the input to the function.
    A: A (V + 1 x sigma + 2) array. For calculating alpha
    y: A (V x 1) array to calculate the outputs.

    returns a (1 x sigma + 1) matrix. Should be the same shape as beta passed in.

    """
    K = cp.diag(alpha.flatten()) # Construct the diagonal matrix.
    sigma = alpha.shape[0]
    V = len(partition)
    partition = partition.reshape([1,V])
    G = np.zeros([sigma, V])
    G[0] = -1
    count = 1
    meme = time.time()
    for i in beta.flatten():
        i = cp.asnumpy(i)
        G[count] = np.apply_along_axis(lambda x, b: -1 if x >= b else 0, 0, partition, i)
        count += 1
    print(f'{time.time() - meme} for gradient')
    G = cp.asarray(G)
    r = A @ alpha - y
    g = 2 * (K @ G @ r)
    Y = 2 * (K @ G @ G.T @ K.T)
    if cp.linalg.det(Y) == 0:
        print('Determinant of 0')
        return -g # Return the gradient if Y is singular
    inv = cp.linalg.inv(Y)
    s = -inv @ g
    return s

def calculate_alpha(partition, beta, y):
    """
    Calculate the alpha that has minimal loss given beta and a partition.

    partition: A V length array representing the points to build on.
    beta: A (1 x sigma) array that represents the current beta setting.
    y: A (V x 1) array representing the target indices.

    Return alpha, which is a (sigma, 1) vector, and A, which is a (V, sigma) matrix
    """
    V = len(partition)
    sigma = beta.shape[1] # How many betas we got???
    y = cp.arange(V).reshape([V,1])
    A = np.zeros([V, sigma + 1]) # Remember that a row in A is [1, xi-B0, (xi - B1) if xi >= B1, ..., xi - B_sigma, if x 0 > = B_sigma]. This is len(beta) + 1 == sigma + 1 + 1
    count = 0
    meme = time.time()
    beta = cp.asnumpy(beta)
    for i in partition:
        row = np.zeros([1, sigma + 1])
        row[0][0] = 1
        row[0][1:] = np.apply_along_axis(lambda b, x: x - b if x > b else 0, 0, beta, i)
        A[count] = row.flatten()
        count += 1
    print(f'{time.time() - meme} seconds for calculate_alpha')
    A = cp.asarray(A)
    inv = cp.linalg.inv(cp.matmul(A.T, A))
    alpha = cp.matmul(cp.matmul(inv, A.T), y) # Alpha is a bar, a0, a1 ,..., a_sigma
    return alpha, A

def find_interval(point, Theta, T_i):
    """
    point is a 1 x dimension array.
    Return the interval that we found the point in. IE a dimension x 2 array.

    Will be of form ((x_0, x_1), (y_0, y_1), (z_0, z_1) ...) for all dimension in dimension.
    """
    dimension = Theta.shape[0]
    lst = []
    for i in range(dimension):
        p = binary_search(point[i], Theta[i], 0, T_i[i] - 1)
        lst.append(p)

    return tuple(lst)

def binary_search(element, array, beginning, end):
    """
    Return the interval of the element that it closest belongs to in an array.
    """
    mid = (beginning + end)//2
    if element < array[beginning]: # End the binary search if the element is less than min
        return array[beginning], array[beginning + 1] # Set to the beginning.
    if element > array[end]: # In this particular case, we have something belonging in the last interval.
        return array[end], array[end + 1]
    if array[mid] <= element < array[mid + 1]:
        return array[mid], array[mid + 1]
    else:
        if element > array[mid] and element >= array[mid + 1]:
            return binary_search(element, array, mid + 1, end)
        else:
            return binary_search(element, array, beginning, mid)

def mapping_list_partition_cells(cells, k):
    """
    Returns a k x F array where F is approximately len(lst)/k equal partition of the list of points
    Also returns the original point list with an attached mapping to them, sorted by mapping

    lst is a N x dimension list of points
    cells is a list of cells that partition the entire dataset
    k an integer representing the number of partitions we should break up the original list into
    """
    parts_with_mapping = mapping_function_cells(cells)

    parts = np.array_split(parts_with_mapping[:, 0], k)
    pdb.set_trace()
    return parts, parts_with_mapping

def mapping_list_partition(lst, Theta, T_i, k):
    """
    Returns a k x F, where F is approximately len(lst)/k equal partition of the list of points.
    Also returns the original point list with an attached mapping to them, sorted by mapping.

    lst is a N x dimension list of points.
    Theta are the border points.
    T_i are the size of the border points.
    """
    # Apply mapping_function on all points given.
    length = lst.shape[0]
    mapping = np.apply_along_axis(mapping_function, 1, lst, Theta, T_i).reshape((length, 1)) # Merge mapping with lst.
    parts_with_mapping = np.concatenate((mapping, lst), axis = 1)
    parts_with_mapping = parts_with_mapping[np.argsort(parts_with_mapping[:,0])] # Sort based on mapping column, and sort the entire list based off that.

    parts = np.array_split(parts_with_mapping[:, 0], k)
    
    return parts, parts_with_mapping
        
def cell_index(point, Theta, T_i):
    """
    point is a 1x dimension length array.
    Theta is a dimension x max(T_i) length array.
    T_i is a 1 x dimension length array.
    Calculates the cell index based on the given point and Theta.
    """
    dimension = Theta.shape[0]
    interval = find_interval(point, Theta, T_i)
    inds = [np.where(Theta[i] == e[0])[0][0] for i, e in enumerate(interval)]
    count = inds[0]
    for i in range(1, dimension):
        count = count * T_i[i] + inds[i]

    return count

def mapping_function_cells(cells):
    """
    Returns the mapping value for all the points inside all the cells.

    Cells is a list of cells
    """
    lst = []
    pdb.set_trace()
    for cell in cells:
        index = cell.mapping
        mins, maxs = cell.bounding_rectangle
        maxs = maxs + 0.01
        dimension = mins.shape[0]
        C_i = 1
        for i in range(dimension):
            C_i *= maxs[i] - mins[i]
        for point in cell.data:
            H_i = 1
            for i in range(dimension):
                H_i *= point[i] - mins[i]
            p = list(point)
            lst.append(np.asarray((index + (H_i/C_i), *p)))
    return np.asarray(lst)


def mapping_function_with_index(ind, point_lower, point, Theta, T_i):
    """
    ind is an integer
    Point_lower is a 1x dimension array
    point is a 1 x dimension array.
    Theta is a dimension x 2 array
    T_i is a 1 x dimension array
    Returns the value of the mapping function on point, but takes in the Lebesgue measure between point_lower and point and adds the given index. 
    """
    dimension = Theta.shape[0]
    interval = find_interval(point_lower, Theta, T_i)
    C_i = 1
    H_i = 1
    for i,e in enumerate(interval):
        C_i *= e[1] - e[0]
        H_i *= point[i] - e[0]
    return ind + (H_i/C_i)

def mapping_function(point, Theta, T_i):
    """
    Point is a 1 x dimension array.
    Theta is a dimension x 2 array
    T_i is a 1 x dimension array
    Returns the value of the mapping function on point, Theta, T_i
    """
    interval = find_interval(point, Theta, T_i)
    ind = cell_index(point, Theta, T_i)
    C_i = 1
    H_i = 1
    for i,e in enumerate(interval):
        C_i *= e[1] - e[0]
        H_i *= point[i] - e[0]

    return ind + (H_i/C_i)

def calculate_border_points(axis, T_i, maximum):
    """
    data is a 1 x N of sorted data along an axis.
    maximum is the maximum number of entries along any axis. Maximum >= T_i
    Return a 1 x T_i + 1 array of calculated values along the border point.
    
    Note that the last border point extends out to infinity IE [arr[T_i], max]. Technically this should be infinite.
    We should not really take into account maximum in calculating cell membership.
    """
    lists = np.array_split(axis, T_i)
    max_element = np.amax(axis)
    res = np.zeros([1, maximum + 1])
    for i, e in enumerate(lists):
        res[0][i] = np.amin(e)
    res[0][T_i] = max_element
    
    return res

def calculate_centroid(rectangle):
    """
    Calculate the centroid of a rectangle given its top and bottom corners.
    """
    dimension = rectangle.shape[1]
    bottom = rectangle[0]
    top = rectangle[1]


def create_cells(data, T_i):
    """
    T_i is a N-d array, where T_i[j] is the partition amount for dimension j.
    data is a N x dimension matrix. IE it contains N points in dimension d.

    return a Theta that represents all the cells. Consists of:
    Theta[0] = border points for x (1 x maximum + 1) tuple
    Theta[1] = border points for y (1 x maximum + 1) tuple
    ...
    Theta[dimension] = border points for dimension <dimension> (1 x maximum + 1) tuple

    """
    dimension = data.shape[1]
    maximum = np.max(T_i)
    res = np.zeros((dimension, maximum + 1))

    for i in range(dimension):
        axis = data[:, i]
        axis = np.sort(axis)
        res[i] = calculate_border_points(axis, T_i[i], maximum)
        # Avoid index out of range when calculating all dimensions == the maximum value for mapping value.
        res[i][-1] += 0.01

    return res

# TODO: Change
def create_shards(params, full_lst, psi, cells=None):
    """
    Create the shards based on the params.

    params: List of tuples for index i, representing F_i(x). In other words, contains alpha, beta, and the resulting calculation of F_i(x).
    full_lst: List of points including mapping in the first column.
    psi: The expected number of items in a shard
    cells: A list of cells that a shard can potentially belong to.

    Returns a list of shards. shards[i] corresponds to the shard prediction function for learned function F_i
    """
    pdb.set_trace()
    shards = []
    D = np.ceil(len(params[0][-1])/psi)
    count = 0
    points = full_lst # Store the mapping for local shards.
    # points = full_lst[:, 1:] # Grab everything but the mapping. We want only points and they are already sorted by mapping.
    for i, e in enumerate(params):
        alpha, beta, f = e
        V = len(f)
        shard_start_id = i * D
        shard_ind = Shard_index(i)
        positions = np.floor(f/psi).flatten()

        # From the predicted positions, add them to the shard.
        for position in positions:
            position = int(position)
            shard_ind.append(position, points[count])
            curr_mapping = points[count][0]
            curr_mapping = int(np.floor(curr_mapping))
            print(curr_mapping)
            cells[curr_mapping].add_shard((i, position))
            count += 1
        shard_ind.calculate()
        shards.append(shard_ind)

    return shards

def create_shards_original(params, full_lst, psi):
    """
    Create the shards based on the params.

    params: List of tuples for index i, representing F_i(x). In other words, contains alpha, beta, and the resulting calculation of F_i(x).
    full_lst: List of points including mapping in the first column.

    Returns a list of shards. shards[i] corresponds to the shard prediction function for learned function F_i
    """
    shards = []
    D = np.ceil(len(params[0][-1])/psi)
    count = 0
    points = full_lst # Store the mapping for local shards.
    # points = full_lst[:, 1:] # Grab everything but the mapping. We want only points and they are already sorted by mapping.
    for i, e in enumerate(params):
        alpha, beta, f = e
        V = len(f)
        shard_start_id = i * D
        shard = {}
        positions = np.floor(f/psi).flatten()
        for position in positions:
            position = int(position)

            if position in shard:
                shard[position].append(points[count])
            else:
                shard[position] = [points[count]]
            count += 1
        shards.append(shard)

    return shards

# TODO: Change

def decompose_query(Theta, bottom_left, top_right):
    """
    Takes a query rectangle and we decompose it into cells that intersect with the cells.

    bottom_left: 1xdimension numpy array representing the bottom left vertex
    top_right: 1xdimension numpy array representing the top right vertex.
    Theta: The breakpoints for all the axis.

    Returns a list of points that overlap with Theta.
    """
    dimension = Theta.shape[0]
    list_of_points = []
    for i in range(dimension):
        points = []
        # Get the mins and maxs for each dimension
        mins = bottom_left.flatten()
        maxs = top_right.flatten()
        lst = [mins[i]] if mins[i] > Theta[i][0] else [Theta[i][0]]

        # Iterate over the dimension
        for j in Theta[i]:
            if mins[i] < j < maxs[i]:
                lst.append(j)

        # Create the points of overlap on dimension i along Theta[i]'s breakpoints
        lst.append(maxs[i])
        
        for j in range(len(lst) - 1):
            points.append((lst[j], lst[j + 1]))
        list_of_points.append(points)

    full_points = dfs(list_of_points, dimension - 1, 0)
    # Consider list(map(lambda x : list(zip(*x)), list(zip(*list_of_points)))
    # Essentially you're trying to take from columns instead of creating pairwise things.
    # full_points = list(zip(list_of_points))
    result = [tuple(zip(*i)) for i in full_points]

    return result

    # full_points = list(map(lambda x : tuple(zip(*x)), list(zip(*full_points))))

def shard_prediction(mapping, index, params, psi):
    """
    mapping is a float representing a point's mapped value.
    index is used in using F_index to calculate the shard prediction
    params is a K x 3 list representing alpha, beta, and F_i values
    """
    alpha, beta, _ = params[index]
    total = alpha[0] # Set to alpha bar.
    for i in range(len(beta)):
        if mapping >= beta[i]:
            total += alpha[i+1] * (mapping-beta[i])
    return np.floor(total/psi)

def shard_get(index, lower, upper, shards, psi, lmap, umap):
    """
    Returns all the data points within a shard at index and with id in the interval [lower, upper] with points with mapping satisfying lmap <= mapping <= upper
    Also returns a set of pages in the form (index, id, page number) that was used to access these points.

    Get data points from a shard at index index. Get from shard ids lower to upper.

    Use lmap and umap to get specific lower mapping and upper mappings and thus which
    pages to access.
    
    index: integer representing the index of the shard_index to access
    lower: integer represenitng the lower shard_id we need to access
    upper: integer representing the upper shard_id we need to access
    shards: the entire shard list containing all the shard_index
    psi: The average number of datapoints per page
    lmap: the lower mapping that we are looking for
    umap: The upper mapping that we are limiting our search for.
    """
    k = []
    page_set = set()
    pages = 0
    shard_ind = shards[index]
    if lower is None and upper is None:
        # Return everything within the shard at index index.
        for i in shard_ind.get_keys():
            data = shard_ind.get(i).get_data()
            for page in range(int(np.ceil(len(data)/psi))):
                page_set.add((index, i, page))
            pages += np.ceil(len(data)/psi)
            k.extend(data)
    elif lower is None:
        upper = int(upper)
        found_end_index = None
        for i in shard_ind.get_keys():
            if i <= upper:
                curr_shard = shard_ind.get(i)
                data = curr_shard.get_data()
                if found_end_index is not None:
                    break
                else:
                    for ind, element in enumerate(data):
                        if element[0] > umap:
                            found_end_index = ind
                            break
                    if found_end_index:
                        k.extend(data[:found_end_index])
                        for page in range(int(np.ceil(found_end_index/psi))):
                            page_set.add((index, i, page))
                        pages += np.ceil(found_end_index/psi)
                    else:
                        k.extend(data)
                        for page in range(int(np.ceil(len(data)/psi))):
                            page_set.add((index, i, page))
                        pages += np.ceil(len(data)/psi)
    elif upper is None:
        lower = int(lower)
        keys = sorted(shard_ind.get_keys())
        max_key = keys[-1]
        i = lower
        found_index = None
        while i <= max_key:
            # Check which page to start at.
            if i in keys:
                data = shard_ind.get(i).get_data()
                start_page = None
                if found_index is None:
                    for ind, element in enumerate(data):
                        if element[0] > lmap:
                            # Start from here.
                            found_index = ind
                            break
            if found_index is not None:
                data = shard_ind.get(i).get_data()
                start_page = int(np.floor(found_index/psi)) # Starting page
                k.extend(data[found_index:])
                end_page = int(np.ceil(len(data)/psi)) 
                for page in range(start_page, end_page):
                    page_set.add((index, i, page))
                pages += np.ceil(len(data)/psi) - start_page 
                found_index = 0 # Reset to zero and never touch it again. This is because once we found the start index, we keep going on all the other relevant shards.

            i += 1
    else:
        upper, lower = int(upper), int(lower)
        for i in range(lower, upper + 1):
            res = []
            if i in shard_ind.get_keys():
                data = shard_ind.get(i).get_data()
                res = [i for i, e in enumerate(data) if lmap <= e[0] <=umap]
                if len(res) == 0: # No mapping was found in this range
                    # print("Hmmm")
                    continue
                else:
                    start = min(res)
                    end = max(res)
                    start_page = int(np.floor(start/psi))
                    end_page = int(np.floor(end/psi))
                    for page in range(start_page, end_page+1):
                        page_set.add((index, i, page))
                    pages += end_page - start_page + 1
                    k.extend(data[start:end + 1])
    return k, pages, page_set

# TODO: Change
def find_points(points, params, shards, M, T_i, psi, Theta):
    """
    points is a list of tuples. These tuples are of form (bottom_left, top_right), where both bottom_left and top_right are 1x2 np arrays.
    params is a K X 3 list representing alpha, beta, and F_i in that order
    shards is a list of dictionaries that represents what each shard[i] sets.
    M is a 1 x M lenght list representing the partition points.
    T_i is a 2 x N length list representing the lengths of Theta.
    psi is a integer representing the expected number of keys falling in a range.
    Theta is the cell array.

    Returns a list of points found that lie within points.
    """
    answer = []
    pages = 0
    page_set = set()
    for i in points:
        #TODO: Maybe care about out of bounds/ignore it?
        lower_left, top_right = i
        lower_mapping = mapping_function(lower_left, Theta, T_i)
        upper_mapping = mapping_function(top_right, Theta, T_i)


        if upper_mapping - np.floor(lower_mapping) > 1:
            upper_mapping = mapping_function_with_index(np.floor(lower_mapping), lower_left, top_right, Theta, T_i)

        lower = binary_search(lower_mapping, M, 0, len(M) - 2) # We remove 2 from length M, because the bin search will return an interval
        upper = binary_search(upper_mapping, M, 0, len(M) - 2)
        index_lower = M.index(lower[0])
        index_upper = M.index(upper[0])
        shard_pred_u = shard_prediction(upper_mapping, index_upper, params, psi)
        shard_pred_l = shard_prediction(lower_mapping, index_lower, params, psi)
        
        # TODO: See if we can create a full shard list, instead of on the fly grabbing the shards that matter.
        # Perhaps that will speed things up??
        if index_upper < index_lower:
            raise Exception("Upper less than lower. Failed")
        if index_upper > index_lower:
            k, p, ps = shard_get(index_lower, shard_pred_l, None, shards, psi, lower_mapping, upper_mapping)
            page_set.update(ps)
            pages += p
            for i in range(index_lower + 1, index_upper):
                ans, p, ps= shard_get(i, None, None, shards, psi, lower_mapping, upper_mapping)
                k.extend(ans)
                page_set.update(ps)
                pages += p
            ans, p, ps = shard_get(index_upper, None, shard_pred_u, shards, psi, lower_mapping, upper_mapping)
            k.extend(ans)
            pages += p
            page_set.update(ps)
        else:
            k, p, ps = shard_get(index_lower, shard_pred_l, shard_pred_u, shards, psi, lower_mapping, upper_mapping) 
            pages += p
            page_set.update(ps)
        answer.extend([tuple(i) for i in k])

    return answer, pages, page_set

def visualize_cells(lst, cells, psi):
    x = lst[:, 0]
    y = lst[:, 1]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(x,y, color='white')

    for cell in cells:
        bottom, top = cell.bounding_rectangle
        ax.add_patch(Rectangle(xy =(bottom[0], bottom[1]), width= top[0] - bottom[0], height = top[1] - bottom[1], fill=False, color='purple'))
        ax.annotate(f'{cell.mapping}', top)
    plt.show()


def visualize(Theta, lst, T_i, params, psi, M, cells):
    x = np.take(lst, 0, 1)
    y = np.take(lst, 1, 1)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for i in range(T_i[0]): # We need to know the length of the T_i
       for j in range(T_i[1]):
           ax.add_patch(Rectangle(xy=(Theta[0][i], Theta[1][j]), width=Theta[0][i+1]-Theta[0][i], height=Theta[1][j+1]-Theta[1][j], fill=False, color='blue'))

    for i in cells:
        bottom_left = i.bounding_rectangle[0]
        top_right = i.bounding_rectangle[1]
        ax.add_patch(Rectangle(xy=(bottom_left[0], bottom_left[1]), width = top_right[0] - bottom_left[0], height = top_right[1] - bottom_left[1], fill=False, color='purple'))
    ax.scatter(x, y, color='red')
    i = 0
    # while i < lst.shape[0]:
    #     mapping = mapping_function(lst[i], Theta, T_i)
    #     ind = M.index(binary_search(mapping, M, 0, len(M) - 2)[0])
    #     shard_pos = shard_prediction(mapping, ind, params, psi)
    #     ax.annotate(f"({ind}, {shard_pos})", lst[i])
    #     i += 1

    plt.show()

def in_query(point, q_rectangle):
    """
    Determine if point belongs in q_rectangle.
    """
    low, up = q_rectangle
    dimension = len(point)
    for i in range(dimension):
        if low[i] <= point[i] <= up[i]:
            continue
        else:
            return False
    return True

def sort_branches(branches, min_dist_ordering=False):
    """
    Sort branches based off of min_dist/min_max_dist.

    If min_dist_ordering is true, sort by min_dist. Otherwise, sort by min_max_dist

    branches a list of the form (min_max_dist, min_dist, object)

    Returns a sorted list in descending order.
    """
    if branches == []:
        return []
    sort = 1 if min_dist_ordering is True else 0
    branches = np.asarray(branches)
    branches = branches[np.argsort(branches[:,sort])]
    return list(branches)

def find_nearest(results, point, nearest, k):
    """
    Given a list of results, find those that are closer to the point and update nearest as we see fit.

    results: List of found points that were found in the region.
    point: The point to look for closest to
    nearest: The current array of closest points.
    """
    dimension = len(point)
    for i in results:
        dist = distance(i[1:], point) 
        dist_entry = np.insert(i[1:], 0, dist)
        if len(nearest) < k:
            nearest.append(dist_entry)
        else:
            nearest = np.asarray(nearest)
            max_dist = np.max(np.take(nearest, 0, 1))
            if dist < max_dist:
                nearest = np.asarray(nearest)
                nearest = np.concatenate((nearest, dist_entry.reshape(1,dimension + 1)), 0)
                nearest = nearest[np.argsort(nearest[:,0])][:k] # Take the closest k elements
            nearest = list(nearest)
    return nearest

# TODO: Change
def KNN_updated(k, point, params, shards, psi, cells):
    """
    Finds the K nearest neighbours to a given point.

    Uses the MINDIST/MINMAXDIST ordering to prune the search set.
    """
    # Generate cells to visit by mindist/minmaxdist.
    timer = []
    timer1 = []
    timer2 = []
    timer3 = []
    nearest = []
    iters = 0
    pages = 0
    branches = []
    page_set = set()
    shard_set = set()

    for i in cells:
        # Force prune to never happen based off of minmaxdist
        # Set min_max_dist to infinity because cells aren't MBRs.
        branches.append((float('inf'), min_dist(point, i.bounding_rectangle), i))
    branches = sort_branches(branches, True)
    branches = prune(branches, nearest, k)

    while len(branches) > 0:
        cell = branches.pop(0)[-1]

        to_visit = []

        # Grab the shards within the cell
        # Sort them based off of the closest bounding rectangle
        # TODO: Maybe instead of adding every bounding rectangle or "closest bounding rectangle", why not just add the rectangle that overlaps with
        # the cell that we are visiting?
        for shard_index in cell.shards:
            ind, id = shard_index
            
            # Perhaps we are revisiting too many shards.
            if shard_index in shard_set:
                continue
            shard_set.add(shard_index)
            shard = shards[ind].get(id)

            # Get all the bounding rectangles associated with this shard. Only "visit" based on the closest min_dist to the point.

            b = time.time()
            entries = np.asarray([(min_max_dist(point, br.rectangle), min_dist(point, br.rectangle), ind, br, shard) for br in shard.get_bounding_rectangles()])
            to_visit.extend(entries[np.argsort(entries[:,1])])
            timer1.append(time.time() - b)

        # pdb.set_trace()
        to_visit = sort_branches(to_visit)
        if len(to_visit) == 0:
            continue

        d = time.time()
        # Do downwards pruning
        to_visit = prune(to_visit, nearest, k)
        timer3.append(time.time() - d)
        
        
        a = time.time()
        # Prune only when nearest changes.
        curr_max_nearest = None
        new_nearest = float('inf')
        if len(nearest) != k:
            curr_max_nearest = 0
        else:
            curr_max_nearest = max([i[0] for i in nearest])
        
        while len(to_visit) > 0:
            #TODO: DO NOT JUST LOOK FOR THE BOUNDING RECTANGLE
            # This bounding rectangle can overlap many different shards, resulting in additional accesses.
            # We need to access the relevant shard ONLY as well as the bounding rectangle around this relevant shard.

            # TODO: USE SHARD_GET INSTEAD OF FIND_POINTS to keep it relevant to the specific shard.
            _, _, index, rect, shard = to_visit.pop(0)
            results = []

            # From the shard, get the pages that are needed to get at least this rectangle.
            get_results, _, ps = shard_get(index, shard.id, shard.id, shards, psi, rect.lower_mapping, rect.upper_mapping)
            
            # Using the pages, we go to each of them if we never visited them. If we already visited them,
            # then results already has taken them into consideration.
            for i in ps:
                if i not in page_set:
                    results.extend(shard.get_page(i[-1], psi))
            page_set.update(ps)

            nearest = find_nearest(results, point, nearest, k)

            # Do pruning amongst the shard set
            to_visit = prune(to_visit, nearest, k)
        new_nearest = float('inf')
        if len(nearest) != k:
            new_nearest = float('inf')
        else:
            new_nearest = max([i[0] for i in nearest])
        timer.append(time.time() - a)
        
        c = time.time()
        # Only prune if nearest changes, or max_nearest has changed
        if len(nearest) == k and new_nearest < curr_max_nearest:
            branches = prune(branches, nearest, k) # Prune the branches based off of increasing nearest.
            iters += 1
        timer2.append(time.time()- c)
    print(f"Took {iters} pruning")
    print(f"Took {sum(timer)} seconds visiting")
    print(f'Took {sum(timer1)} generating the shard_list')
    print(f'Took {sum(timer2)} pruning branches')
    print(f'Took {sum(timer3)} pruning and sorting to_visit')
    return nearest, len(page_set)

def KNN(k, delta, point, Theta, params, shards, M, T_i, psi):
    """
    Calculates KNN for point point, along with page accesses.

    k: An int representing how many points we want to return
    delta: A delta that we want to search in.
    point: The point to consider.
    Theta: The breakpoints along every axis
    params: The calculated parameters for LISA
    shards: The points contained in their shards
    M: the partitions we have
    T_i: The length of Theta for each dimension
    psi: How many points are in a shard.
    """
    results = []
    page_set = set()
    while True:
        # Craft the rectangle
        dimension = Theta.shape[0]
        bottom_left = np.zeros(dimension)
        top_right = np.zeros(dimension)
        for i in range(dimension):
            bottom_left[i] = point[i] - delta
            top_right[i] = point[i] + delta

        # Decompose the rectangle, then get the results based on the range query.
        q_rectangles = decompose_query(Theta, bottom_left, top_right)
        _, _, ps = find_points(q_rectangles, params, shards, M, T_i, psi, Theta)
        if page_set != ps:
            results = []
            for i in ps:
                results.extend(shards[i[0]].get(i[1]).get_page(i[-1], psi))
            page_set = ps

        found_points = [res[1:] for res in results if math.sqrt(distance(point, res[1:])) <= delta]
        answer = []
        for i in found_points:
            answer.append((distance(i, point), i[0], i[1]))
        if len(answer) == 0:
            delta += 1
            continue
        answer = np.asarray(answer)
        answer = answer[np.argsort(answer[:,0])]
        if answer.shape[0] < k:
            delta += 1
            continue
        else:
            answer = np.asarray(answer)[:k] # Take the first k elements.
            return answer, page_set

def distance(point1, point2):
    """
    Calculates the square distance between two points

    point1 is a 1x dimension array.
    point2 is a 1x dimension array.
    """
    dimension = len(point1)
    total = 0
    for i in range(dimension):
        total += (point1[i] - point2[i])**2

    return total

def mergeSort(arr):
    if len(arr) > 1:
 
         # Finding the mid of the array
        mid = len(arr)//2
 
        # Dividing the array elements
        L = arr[:mid]
 
        # into 2 halves
        R = arr[mid:]
 
        # Sorting the first half
        mergeSort(L)
 
        # Sorting the second half
        mergeSort(R)
 
        i = j = k = 0
 
        # Copy data to temp arrays L[] and R[]
        while i < len(L) and j < len(R):
            if compare_bounding_rectangles(L[i], R[j]) < 0:
                arr[k] = L[i]
                i += 1
            else:
                arr[k] = R[j]
                j += 1
            k += 1
 
        # Checking if any element was left
        while i < len(L):
            arr[k] = L[i]
            i += 1
            k += 1
 
        while j < len(R):
            arr[k] = R[j]
            j += 1
            k += 1

def compare_bounding_rectangles(rect1, rect2):
    """
    Compares two bounding rectangles.

    rect1 is greater than rect2 if the maximum corner of rect1 is greater than the minimum corner rect2
    IE if rect1's top_right = (x1, x2, ... , xn) and rect2's minimum corner is (y1, y2, ..., yn) then
    rect2 > rect1

    rect1 is a 2 x n tuple. This is a python tuple and not a numpy array
    rect2 is a 2 x n tuple. This is a python tuple and not a numpy array

    The rectangle tuple is of form (minimum_corner, maximum_corner)
    """
    dimension = len(rect1[0])

    # Make sure they are the same dimension
    assert len(rect1[0]) == len(rect2[0])

    # Compare the rectangles
    maximum = np.asarray(rect1[1])
    minimum = np.asarray(rect2[0])

    if (maximum > minimum).all():
        return 1
    else:
        return -1

def generate_qrects(mins, maxs):
    """
    Generate query rectangles based on mins and maxs

    mins is a [1x dimension array] where mins[i] == minimum value on dimension i
    maxs is a [1 x dimension array] where maxs[i] == maximum value on dimension i
    """
    dimension = len(mins)
    lst = generate_numbers(np.min(mins), np.max(maxs), 100, dimension)
    # Calculate the max lengths along each dimension
    max_lengths = []
    for i in range(dimension):
        max_lengths.append(0.25 * (maxs[i] - mins[i]))
    rng = np.random.default_rng()

    k = []
    for i in lst:
        curr_tuple = []
        for j in range(dimension):
            length = rng.uniform(0, max_lengths[j])
            curr_tuple.append(i[j] + length)
        k.append((np.asarray(i), np.asarray(curr_tuple)))
    return k

def test_synthetics():
    open("lisa_synthetic_time.txt", 'w')
    for amount in [1000, 4000, 8000, 16000, 32000, 64000]:
        lst = pickle.load(open(f'synthetic_{amount}.dump', 'rb'))
        T_i = [100, 100]
        Theta = create_cells(lst, T_i)
        partitions, full_lst = mapping_list_partition(lst, Theta, T_i, 10)
        M = [partitions[0][0]]
        for i in partitions:
            M.append(i.max())
        psi = 50
        a = time.time()
        params = train(partitions, 3, psi) # Train the partitions with 2 breakpoints. Tunable hyperparameter. IE sigma + 1 == second parameter.
        a = time.time() - a
        bounding_rectangles = decompose_query(Theta, np.asarray((Theta[0][0], Theta[1][0])), np.asarray((Theta[0][-1], Theta[1][-1])))
        cells = []
        for i in range(len(bounding_rectangles)):
            cells.append(Cell(i, bounding_rectangles[i]))

        shards = create_shards(params, full_lst, psi, cells)
    
        for i in cells:
            print(i.mapping, i.shards)
        for shard_ind in shards:
            for shard_id in shard_ind.get_keys():
                shard = shard_ind.get(shard_id)
                print(shard.lower_mapping, shard.upper_mapping, len(shard.bounding_rectangles), (shard_ind.get_id(), shard.id))

        min_x = np.min(lst[:,0])
        min_y = np.min(lst[:,1])
        max_x, max_y = np.max(lst[:,0]), np.max(lst[:,1])
        # q_rects = generate_qrects(min_x, min_y, max_x, max_y)
        # pickle.dump(q_rects, open(f"synthetic_qrects_{amount}.dump", "wb"))
        q_rects = pickle.load(open(f"synthetic_qrects_{amount}.dump", "rb"))

        total_p = 0
        count = 0
        f = open("lisa_synthetic_query.txt", "w")
        f.write("Took {a} time training.\n")
        f.close()
        for i in q_rects:
            points = decompose_query(Theta, i[0], i[1])
            return_results, pages, page_set = find_points(points, params, shards, M, T_i, psi, Theta)
            final_results = {tuple(point[1:]) for point in return_results if in_query(point[1:], i)} # Remove mapping from the points.
            actual = [point for point in lst if in_query(point, i)]
            total_p += len(page_set)

            with open("lisa_synthetic_query.txt", "a") as output:
                output.write(f"{amount} Actual: {len(actual)}\n")
                output.write(f"Results: {len(final_results)}\n")
                if len(final_results) != len(actual):
                    output.write("OOF\n")
                output.write(f"{len(page_set)}\n")
                output.write(f"Average page lookup {total_p/100}.\n")
                

            print(f"Actual: {len(actual)}")
            print(f"Results: {len(final_results)}")
            print(len(page_set))
            count += 1
        print(f"Average page lookup {total_p/100}.")

        # KNN_random_points = generate_numbers(0, 8000, 100)
        # # pickle.dump(KNN_random_points, open(f"synthetic_qpoints_{amount}.dump", "wb"))
        KNN_random_points = pickle.load(open(f"synthetic_qpoints_{amount}.dump", "rb"))
        timer = []
        
        for K in [50]:
        # for K in [1, 5, 10, 50, 100, 500]:
            pages = 0
            print(f'K: {pages}')
            for point in KNN_random_points:
                a = time.time()
                nearest, p = KNN_updated(K, point, params, shards, psi, cells)
                timer.append(time.time() - a)
                nearest = np.asarray(nearest)
                nearest = nearest[np.argsort(nearest[:, 0])]
                pages += p
                actual = np.asarray([np.asarray((distance(point, i), i[0], i[1])) for i in lst])
                actual = actual[np.argsort(actual[:,0])][:K]
                if not (actual[:, 1:] == np.asarray(nearest)[:, 1:]).all():
                    print(actual)
                    print(nearest)
                    raise Exception(amount, K)
                # print(actual[:,1:] == np.asarray(nearest)[:,1:])
                print(f"Point: {point}")
                print(f"Neighbours: {np.asarray(nearest)}")
                print(f"Pages: {p}")
            with open("lisa_synthetic_time.txt", 'a') as output:
                output.write(f"Average pages for {K} on synthetic points {amount}: {pages/100}.\nAverage time: {sum(timer)/100}\n")
            print(f"Average pages for {K}: {pages/100}")

        # for K in [50]:
        #     pages = 0
        #     count = 0
        #     timer = []

        #     for point in KNN_random_points:
        #     # for point in [KNN_random_points[9]]:
        #         print(K)
        #         count += 1
        #         a = time.time()
        #         ans, k_pages = KNN(K, 1, point, Theta, params, shards, M, T_i, psi)
        #         timer.append(time.time() - a)
        #         pages += len(k_pages)
        #         actual = np.asarray([np.asarray((distance(point, i), i[0], i[1])) for i in lst])
        #         actual = actual[np.argsort(actual[:,0])][:K]
        #         if not (actual[:, 1:] == np.asarray(ans)[:, 1:]).all():
        #             print(actual)
        #             print(ans)
        #             pdb.set_trace()
        #             KNN(K, 1, point, Theta, params, shards, M, T_i, psi)
        #             raise Exception(amount, count)
        #             print(f"Neighbours: {np.asarray(ans)}")

        #         print(f"Point: {point}")
        #         print(f"Neighbours: {np.asarray(ans)}")
        #     print(f"Average pages: {pages/100}")
        #     with open("lisa_synthetic.txt", 'a') as output:
        #         output.write(f'Average pages for {K} on synthetic points {amount} using delta: {pages/100}\n. Average time is {sum(timer)/100}.\n')

def test_nd():
    open('lisa_synthetic_nd.txt', 'w')
    values = {3: [10, 10, 10], 4:[ 8, 8, 8, 8], 5: [7, 7, 7, 7, 7], 6: [6, 6, 6, 6, 6, 6]}
    for dimension in [6]:
    # for dimension in [3, 4, 5, 6]:
        for amount in [32000]:
        # for amount in [1000, 4000, 8000, 16000, 32000, 64000]:
            lst = pickle.load(open(f'synthetic_{amount}_{dimension}d.dump', 'rb'))
            T_i = values[dimension]
            Theta = create_cells(lst, T_i)
            partitions, full_lst = mapping_list_partition(lst, Theta, T_i, 10)

            M = [partitions[0][0]]
            for i in partitions:
                M.append(i.max())
            psi = 50
            a = time.time()
            params = train(partitions, 3, psi)
            a = time.time() - a

            bounding_rectangles = decompose_query(Theta, Theta[:, 0], Theta[:, -1])
            cells = []
            for i in range(len(bounding_rectangles)):
                cells.append(Cell(i, bounding_rectangles[i]))

            shards = create_shards(params, full_lst, psi, cells)
            mins = []
            maxs = []
            for i in range(dimension):
                mins.append(np.min(lst[:,i]))
                maxs.append(np.max(lst[:,i]))
            with open("lisa_synthetic_nd.txt", 'a') as output:
                output.write(f"Dimension {dimension}\nTook {a} time to train.\n")

            q_points = pickle.load(open(f'synthetic_qpoints_{dimension}d.dump', 'rb'))
            for K in [500]:
            # for K in [1, 5, 10, 50, 100, 500]:
                pages = 0
                # for point in [q_points[0]]:
                for point in q_points:
                    a = time.time()
                    nearest, p = KNN_updated(K, point, params, shards, psi, cells)
                    print(f'Took {time.time() - a} seconds to KNN')
                    nearest = np.asarray(nearest)
                    nearest = nearest[np.argsort(nearest[:, 0])]
                    pages += p
                    actual = np.asarray([np.asarray((distance(point, i), *i)) for i in lst])
                    actual = actual[np.argsort(actual[:,0])][:K]
                    
                    if not (actual[:, 1:] == np.asarray(nearest)[:, 1:]).all():
                        print(actual)
                        print(nearest)
                        raise Exception(amount, K)
                    print(f"Point: {point}")
                    print(f"Neighbours: {np.asarray(nearest)}")
                    print(f"Pages: {p}")
                with open("lisa_synthetic_nd.txt", 'a') as output:
                    output.write(f"Average pages for {K} on synthetic points {amount} for dimension {dimension}D: {pages/100}.\n")
                
                print(f"Average pages for {K}: {pages/100}")
                


def test_3d():
    # Generate a 100 3d points.
    lst = generate_numbers(0, 100, 100, 3)
    # pickle.dump(lst, open("3d_lisa.dump", "wb"))
    # lst = pickle.load(open('3d_lisa.dump', 'rb'))
    dimension = 3
    T_i = [4, 4, 4]
    Theta = create_cells(lst, T_i)

    partitions, full_lst = mapping_list_partition(lst, Theta, T_i, 3)

    M = [partitions[0][0]]
    for i in partitions:
        M.append(i.max())

    psi = 50
    params = train(partitions, 3, psi)
    
    # Get the bounding rectangles.
    # Send in all the minimum values of all Thetas, and the maximum values of all Thetas as bottom_left and top_right respectively
    pdb.set_trace()
    bounding_rectangles = decompose_query(Theta, Theta[:, 0], Theta[:, -1])
    cells = []
    for i in range(len(bounding_rectangles)):
        cells.append(Cell(i, bounding_rectangles[i]))
    pdb.set_trace()

    # Create shards to store the data.   
    shards = create_shards(params, full_lst, psi, cells)
    mins = []
    maxs = []
    for i in range(dimension):
        mins.append(np.min(lst[:,i]))
        maxs.append(np.max(lst[:,i]))

    q_rects = pickle.load(open('3d_lisa_qrects.dump', 'rb'))


    total_p = 0
    count = 0
    # i = q_rects[8]
    # pdb.set_trace()
    # points = decompose_query(Theta, i[0], i[1])
    # _, pages, page_set = find_points(points, params, shards, M, T_i, psi, Theta)
    # return_results = []
    # for page in page_set:
    #     shard_ind = shards[page[0]]
    #     return_results.extend(shard_ind.get(page[1]).get_page(page[-1], psi))
    # final_results = [tuple(point[1:]) for point in return_results if in_query(point[1:], i)]
    # actual = [tuple(point) for point in lst if in_query(point, i)]
    # return
    for i in q_rects:
        print(f'current query {count}')
        points = decompose_query(Theta, i[0], i[1])
        _, pages, page_set = find_points(points, params, shards, M, T_i, psi, Theta)
        return_results = []
        for page in page_set:
            shard_ind = shards[page[0]]
            return_results.extend(shard_ind.get(page[1]).get_page(page[-1], psi))
        final_results = [tuple(point[1:]) for point in return_results if in_query(point[1:], i)] # Remove mapping from the points.
        actual = [tuple(point) for point in lst if in_query(point, i)]
        if len(actual) != len(final_results):
            print(f"Actual: {actual}\nRealized: {final_results}")
        total_p += len(page_set)
        print(f"Results: {len(final_results)}")
        print(len(page_set))
        count += 1
    print(f"Average page lookup {total_p/100}.")
    print("Nice")

    K = 10
    pages = 0
    q_points = generate_numbers(0, 100, 100, 3)
    for point in q_points:
        nearest, p = KNN_updated(K, point, Theta, params, shards, M, psi, cells)
        pages += p
        print(f'Pages {p}\nNeighbours: {nearest}')
        actual = np.asarray([np.asarray((distance(point, i), *i)) for i in lst])
        actual = actual[np.argsort(actual[:,0])][:K]
        if not (actual[:, 1:] == np.asarray(nearest)[:, 1:]).all():
            print(actual)
            print(nearest)
            raise Exception(amount, K)
    print(f"Average Pages: {pages/100}")


    
def test_1000():
    lst = pickle.load(open('data_1000.dump', 'rb'))
    # lst = generate_numbers(0, 100, 1000, 3)
    psi = 50
    T_i = [5,5]
    dimension = 2
    mins, maxs, data = kd.get_leaves(lst, 15) # Should have a total of 32 leaves

    # distances = [np.sqrt(distance(i, [0] * dimension)) for i in maxs]
    # sorted_indices = np.argsort(distances)
    # pdb.set_trace()

    # mins = mins[sorted_indices]
    # maxs = maxs[sorted_indices]
    # data = [data[i] for i in sorted_indices]

    bounding_rects = []
    indices = {}
    cells = []

    for i in range(len(mins)):
        rect = (tuple(mins[i]), (tuple(maxs[i])))
        bounding_rects.append(rect)
        indices[rect] = i

    pdb.set_trace()

    # Sort based off of monotonicity condition.
    pdb.set_trace()
    bounding_rects_sorted = bounding_rects.copy()
    mergeSort(bounding_rects_sorted)
    # bounding_rects_sorted = sorted(bounding_rects, key=cmp_to_key(compare_bounding_rectangles))
    k = np.asarray([indices[i] for i in bounding_rects_sorted])

    mins = mins[k]
    maxs = maxs[k]
    data = [data[i] for i in k]
    
    # Create the cells
    for i in range(len(mins)):
        cell = Cell(i, (mins[i], maxs[i]))
        for j in data[i]:
            cell.add_data(j)
        cells.append(cell)
        print(f'Cell: {i}, data: {len(cell.data)}')

    # Check cells:
    for i in range(len(mins)-1, -1, -1):
        br1 = cells[i].bounding_rectangle
        for j in range(i-1,-1, -1):
            br2 = cells[j].bounding_rectangle
            if (br1[0] < br2[1]).all():
                print(i, j)
                # raise Exception(f'cell {i} is greater than cell {j}')

    partitions, full_lst = mapping_list_partition_cells(cells, 3)
    print(bounding_rects)

    # visualize_cells(lst, cells, psi)

    psi = 50
    params = train(partitions, 3, psi) # Train the partitions with 2 breakpoints. Tunable hyperparameter. IE sigma + 1 == second parameter.

    shards = create_shards(params, full_lst, psi, cells)
    for i in cells:
        print(i.mapping, i.shards)

    
    # visualize(Theta, lst, T_i,params, psi, M, cells )
    # pdb.set_trace()
    for shard_ind in shards:
        for shard_id in shard_ind.get_keys():
            shard = shard_ind.get(shard_id)
            print(shard.lower_mapping, shard.upper_mapping, len(shard.bounding_rectangles), (shard_ind.get_id(), shard.id), len(shard.get_data()))
    
    KNN_random_points = generate_numbers(0, 100, 100, 2)
    K = 100
    pages = 0
    amount = 0
    for point in KNN_random_points:
        nearest, p = KNN_updated(K, point, params, shards, psi, cells)
        nearest = np.asarray(nearest)
        nearest = nearest[np.argsort(nearest[:, 0])]
        pages += p
        actual = np.asarray([np.asarray((distance(point, i), *i)) for i in lst])
        actual = actual[np.argsort(actual[:,0])][:K]
        amount += 1
        
        if not (actual[:, 1:] == np.asarray(nearest)[:, 1:]).all():
            print(actual)
            print(nearest)
            raise Exception(amount, K)
        print(f"Point: {point}")
        print(f"Neighbours: {np.asarray(nearest)}")
        print(f"Pages: {p}")
    
    print(f"Average pages for {K}: {pages/100}")

    # min_x = np.min(lst[:,0])
    # min_y = np.min(lst[:,1])
    # max_x, max_y = np.max(lst[:,0]), np.max(lst[:,1])
    # pdb.set_trace()
    # # q_rects = generate_qrects(min_x, min_y, max_x, max_y)
    # # pickle.dump(q_rects, open("query_rectangles_long_beach.dump", 'wb'))
    # 
    # 
    # q_rects = pickle.load(open("query_rectangles_100_100x100.dump", "rb"))
    # total_p = 0
    # count = 0
    # for i in q_rects:
    #     points = decompose_query(Theta, i[0], i[1])
    #     return_results, pages, page_set = find_points(points, params, shards, M, T_i, psi, Theta)
    #     final_results = {tuple(point[1:]) for point in return_results if in_query(point[1:], i)} # Remove mapping from the points.
    #     total_p += len(page_set)
    #     print(f"Results: {len(final_results)}")
    #     print(len(page_set))
    #     count += 1
    # print(f"Average page lookup {total_p/100}.")

    # 
    # KNN_random_points = pickle.load(open('qpoints_100.dump', 'rb'))
    # point = KNN_random_points[-2]
    # pages = 0
    # k = 10
    # pdb.set_trace()
    # nearest, p = KNN_updated(k, point, Theta, params, shards, M, psi, cells)

    # for point in KNN_random_points:
    #     nearest, p = KNN_updated(k, point, Theta, params, shards, M, psi, cells)
    #     pages += p
    #     print(f"Point: {point}")
    #     print(f"Neighbours: {np.asarray(nearest)}")
    #     print(f"Pages: {p}")
    # print(f"Average pages: {pages/100}")

def test_images():
    lst = pickle.load(open('6d_cifar_100', 'rb'))
    T_i = [6, 6, 6, 6, 6, 6]
    # params, shards, M, T_i, psi, Theta, cells = pickle.load(open('lisa_images_10bp_Ti4.obj', 'rb'))
    # params, shards, M, T_i, psi, Theta, cells = pickle.load(open('lisa_images_cifar_100.obj', 'rb'))
    pdb.set_trace()
    Theta = create_cells(lst, T_i)

    partitions, full_lst = mapping_list_partition(lst, Theta, T_i, 10)
    # print([len(shards[0].get(i).data) for i in shards[0].get_keys()])
    pdb.set_trace()

    M = [partitions[0][0]]
    for i in partitions:
       M.append(i.max())
    psi = 50
    pdb.set_trace()
    params = train(partitions, 10, psi)

    bounding_rectangles = decompose_query(Theta, Theta[:, 0], Theta[:, -1])
    cells = []
    for i in range(len(bounding_rectangles)):
        cells.append(Cell(i, bounding_rectangles[i]))

    shards = create_shards(params, full_lst, psi, cells)
    #pickle.dump([params, shards, M, T_i, psi, Theta, cells], open('lisa_images_10bp.obj', 'wb'))
    #pickle.dump([params, shards, M, T_i, psi, Theta, cells], open('lisa_images_10bp_Ti6.obj', 'wb'))
    pickle.dump([params, shards, M, T_i, psi, Theta, cells], open('lisa_images_cifar_100.obj', 'wb'))
    pdb.set_trace()
    q_points = pickle.load(open('qpoints_images.dump', 'rb'))
    open("lisa_images_cifar_100_results.txt", 'wb')
    for K in [1, 5, 10, 50, 100, 500]:
        pages = 0
        timer = 0
        amount = 0
        dimension = 6
        # for point in [q_points[2]]:
        #    pdb.set_trace()
        for point in q_points:
            a = time.time()
            nearest, p = KNN_updated(K, point, params, shards, psi, cells)
            timer += time.time() - a
            print(f'Took {time.time() - a} seconds to KNN')
            nearest = np.asarray(nearest)
            nearest = nearest[np.argsort(nearest[:, 0])]
            pages += p
            actual = np.asarray([np.asarray((distance(point, i), *i)) for i in lst])
            actual = actual[np.argsort(actual[:,0])][:K]
            amount += 1
            
            if not (actual[:, 1:] == np.asarray(nearest)[:, 1:]).all():
                print(actual)
                print(nearest)
                raise Exception(amount, K)
            print(f"Point: {point}")
            print(f"Neighbours: {np.asarray(nearest)}")
            print(f"Pages: {p}")
        with open("lisa_images_cifar_100_results.txt", 'a') as output:
            output.write(f"Average pages for {K} on synthetic points {amount} for dimension {dimension}D: {pages/100}.\nTook {timer/100} seconds.\n")
        
        print(f"Average pages for {K}: {pages/100}")

def test(in_file, point_file, out_file, dump_file):
    lst = pickle.load(open(in_file, 'rb'))
    T_i = [6, 6, 6, 6, 6, 6]
    open(out_file,'w')
    Theta = create_cells(lst, T_i)

    partitions, full_lst = mapping_list_partition(lst, Theta, T_i, 10)
    M = [partitions[0][0]]
    for i in partitions:
       M.append(i.max())
    psi = 50
    bounding_rectangles = decompose_query(Theta, Theta[:, 0], Theta[:, -1])
    cells = []
    for i in range(len(bounding_rectangles)):
        cells.append(Cell(i, bounding_rectangles[i]))
    a = time.time()
    params = train(partitions, 10, psi) # Train the partitions with 2 breakpoints. Tunable hyperparameter. IE sigma + 1 == second parameter.
    with open(out_file, 'a') as out:
        out.write(f"Took {time.time() - a} to train.\n")
    shards = create_shards(params, full_lst, psi, cells)
    pickle.dump([params, shards, M, T_i, psi, cells], open(dump_file, 'wb'))

    q_points = pickle.load(open(point_file, 'rb'))
    for K in [1, 5, 10, 50, 100, 500]:
        pages = 0
        for point in q_points:
            nearest, p = KNN_updated(K, point, params, shards, psi, cells)
            pages += p
            actual = np.asarray([np.asarray((distance(point, i), *i)) for i in lst])
            actual = actual[np.argsort(actual[:,0])][:K]
            print(f"Point: {point}")
            print(f"Neighbours: {np.asarray(nearest)}")
            print(f"Actual: {actual}")
            print(f"Pages: {p}")
            if not (actual[:, 1:] == np.asarray(nearest)[:, 1:]).all():
                print(actual)
                print(nearest)
                out.write("BIG OOF at {amount} {K}")

        with open(out_file, "a") as out:
            out.write(f"Average pages for {K}: {pages/100}.\n")
        print(f"Average pages: {pages/100}")

def test_long_beach():
    lst = pickle.load(open('LB.dump', 'rb'))
    params, shards, M, T_i, psi, Theta, cells = pickle.load(open("long_beach_lisa_10bp.obj", 'rb'))
    open("lisa_long_beach_query.txt", 'w')

    # TODO: http://sid.cps.unizar.es/projects/ProbabilisticQueries/datasets/ Long beach dataset.
    T_i = [100, 100]

    Theta = create_cells(lst, T_i)
    pdb.set_trace()

    bounding_rectangles = decompose_query(Theta, np.asarray((Theta[0][0], Theta[1][0])), np.asarray((Theta[0][-1], Theta[1][-1])))
    cells = []
    for i in range(len(bounding_rectangles)):
        cells.append(Cell(i, bounding_rectangles[i]))

    pdb.set_trace()

    partitions, full_lst = mapping_list_partition(lst, Theta, T_i, 10) # Create 10 equal length partitions of the mapping space.

    M = [partitions[0][0]]
    for i in partitions:
        M.append(i.max())
    psi = 50
    params = train(partitions, 10, psi) # Train the partitions with 2 breakpoints. Tunable hyperparameter. IE sigma + 1 == second parameter.
    shards = create_shards(params, full_lst, psi, cells)
    pickle.dump([params, shards, M, T_i, psi, Theta, cells], open("long_beach_lisa_10bp.obj", 'wb'))

    min_x = np.min(lst[:,0])
    min_y = np.min(lst[:,1])
    max_x, max_y = np.max(lst[:,0]), np.max(lst[:,1])
    # pdb.set_trace()
    # q_rects = generate_qrects(min_x, min_y, max_x, max_y)
    q_rects = pickle.load(open("query_rectangles_long_beach.dump", 'rb'))
    # pickle.dump(q_rects, open("query_rectangles_long_beach.dump", 'wb'))
    
    total_p = 0
    count = 0
    i = q_rects[69]
    # i = np.asarray(((9927, 485), (9950, 505)))
    # pdb.set_trace()
    # points = decompose_query(Theta, i[0], i[1])
    # return_results, pages, page_set = find_points(points, params, shards, M, T_i, psi, Theta)
    # return_results = []
    # for page in page_set:
    #     shard_ind = shards[page[0]]
    #     shard = shard_ind.get(page[1])
    #     return_results.extend(shard_ind.get(page[1]).get_page(page[-1], psi))
    # actual = [tuple(point) for point in lst if in_query(point, i)]
    # final_results = [tuple(point[1:]) for point in return_results if in_query(point[1:], i)]
    # for i in actual:
    #     if i not in final_results:
    #         print(i)
    # for k in final_results:
    #     if k not in actual:
    #         print("OOF")

    # visualize(Theta, lst, T_i,params, psi, M, cells )
    # pdb.set_trace()
    # print("YIKES")

    for i in q_rects:
        points = decompose_query(Theta, i[0], i[1])
        _, pages, page_set = find_points(points, params, shards, M, T_i, psi, Theta)
        return_results = []
        for page in page_set:
            shard_ind = shards[page[0]]
            return_results.extend(shard_ind.get(page[1]).get_page(page[-1], psi))
        final_results = [tuple(point[1:]) for point in return_results if in_query(point[1:], i)] # Remove mapping from the points.
        actual = [tuple(point) for point in lst if in_query(point, i)]
        total_p += len(page_set)
        with open("lisa_long_beach_query.txt", "a") as output:
            output.write(f"{count} Results: {len(final_results)}\n")
            output.write(f"Actual: {len(actual)}\n")
        print(f"Results: {len(final_results)}")
        print(pages)
        count += 1
    print(f"Average page lookup {total_p/100}.")
    
    max_point = max(max_x, max_y)
    min_point = min(min_x, min_y)
    pdb.set_trace()
    KNN_random_points = pickle.load(open('qpoints_LB.dump', 'rb'))
    # KNN_random_points = generate_numbers(min_point, max_point, 100)
    # pickle.dump(KNN_random_points, open('qpoints_LB.dump','wb'))

    f = open("lisa_long_beach_results.txt", "w")
    f.close()
    for K in [1, 5, 10, 50, 100, 500]:
        pages = 0
        for point in KNN_random_points:
            nearest, p = KNN_updated(K, point, Theta, params, shards, M, psi, cells)
            pages += p
            actual = np.asarray([np.asarray((distance(point, i), i[0], i[1])) for i in lst])
            actual = actual[np.argsort(actual[:,0])][:K]
            print(f"Point: {point}")
            print(f"Neighbours: {np.asarray(nearest)}")
            print(f"Actual: {actual}")
            print(f"Pages: {p}")

        with open("lisa_long_beach_results.txt", "a") as out:
            out.write(f"Average pages for {K}: {pages/100}.\n")
            if not (actual[:, 1:] == np.asarray(nearest)[:, 1:]).all():
                print(actual)
                print(nearest)
                out.write("BIG OOF at {amount} {K}")
        print(f"Average pages: {pages/100}")



    # Real query time.
    # print("TESTING")
    # KNN_random_points = generate_numbers(min_point, max_point, 100)
    # pickle.dump(KNN_random_points, open('qpoints_LB.dump','wb'))
    # KNN_random_point = pickle.load(open('qpoints_LB.dump', 'rb'))
    # # KNN_random_points = pickle.load(open('qpoints_100.dump', 'rb'))
    # pdb.set_trace()
    # delta = 1
    # # delta = average_dist
    # for p in KNN_random_points:
    #     delta = 1
    #     pages = 0
    #     while True:
    #         ans, k_pages = KNN(K, delta, p, Theta, params, shards, M, T_i, psi)
    #         if len(ans)> 1:
    #             print(ans, delta)
    #         if ans.shape[0] < K:
    #             delta += 1
    #             # mult_factor = 2
    #             # if ans.shape[0] > 0:
    #             #     mult_factor = math.sqrt(K/ans.shape[0])
    #             # delta = delta * mult_factor
    #         else:
    #             average_pages += k_pages

    #             print(f"Point {p}")
    #             print(f'Neighbours {ans}\nPages: {k_pages}')
    #             break
    # print(average_pages/100)
    # visualize(Theta, lst, T_i,params, psi, M )

class kd_tree_implementation:
    '''
    Use the kd_tree to generate bounding boxes.
    '''
    def test_1000():
        lst = pickle.load(open('data_1000.dump', 'rb'))
        # lst = generate_numbers(0, 100, 1000, 3)
        psi = 50
        T_i = [5,5]
        dimension = 2
        mins, maxs, data = kd.get_leaves(lst, 15) # Should have a total of 32 leaves
    
        bounding_rects = []
        indices = {}
        cells = []
    
        for i in range(len(mins)):
            rect = (tuple(mins[i]), (tuple(maxs[i])))
            bounding_rects.append(rect)
            indices[rect] = i
    
        pdb.set_trace()
    
        # Sort based off of monotonicity condition.
        pdb.set_trace()
        bounding_rects_sorted = bounding_rects.copy()
        mergeSort(bounding_rects_sorted)
        k = np.asarray([indices[i] for i in bounding_rects_sorted])
    
        mins = mins[k]
        maxs = maxs[k]
        data = [data[i] for i in k]
        
        # Create the cells
        for i in range(len(mins)):
            cell = Cell(i, (mins[i], maxs[i]))
            for j in data[i]:
                cell.add_data(j)
            cells.append(cell)
            print(f'Cell: {i}, data: {len(cell.data)}')
    
        # Check cells:
        for i in range(len(mins)-1, -1, -1):
            br1 = cells[i].bounding_rectangle
            for j in range(i-1,-1, -1):
                br2 = cells[j].bounding_rectangle
                if (br1[0] < br2[1]).all():
                    print(i, j)
                    # raise Exception(f'cell {i} is greater than cell {j}')
    
        partitions, full_lst = mapping_list_partition_cells(cells, 3)
        print(bounding_rects)
    
        # visualize_cells(lst, cells, psi)
    
        psi = 50
        params = train(partitions, 3, psi) # Train the partitions with 2 breakpoints. Tunable hyperparameter. IE sigma + 1 == second parameter.
    
        shards = create_shards(params, full_lst, psi, cells)
        for i in cells:
            print(i.mapping, i.shards)
    
        
        # visualize(Theta, lst, T_i,params, psi, M, cells )
        # pdb.set_trace()
        for shard_ind in shards:
            for shard_id in shard_ind.get_keys():
                shard = shard_ind.get(shard_id)
                print(shard.lower_mapping, shard.upper_mapping, len(shard.bounding_rectangles), (shard_ind.get_id(), shard.id), len(shard.get_data()))
        
        KNN_random_points = generate_numbers(0, 100, 100, 2)
        K = 100
        pages = 0
        amount = 0
        for point in KNN_random_points:
            nearest, p = KNN_updated(K, point, params, shards, psi, cells)
            nearest = np.asarray(nearest)
            nearest = nearest[np.argsort(nearest[:, 0])]
            pages += p
            actual = np.asarray([np.asarray((distance(point, i), *i)) for i in lst])
            actual = actual[np.argsort(actual[:,0])][:K]
            amount += 1
            
            if not (actual[:, 1:] == np.asarray(nearest)[:, 1:]).all():
                print(actual)
                print(nearest)
                raise Exception(amount, K)
            print(f"Point: {point}")
            print(f"Neighbours: {np.asarray(nearest)}")
            print(f"Pages: {p}")
        
        print(f"Average pages for {K}: {pages/100}")

    def test_images():
        lst = pickle.load(open('6d_cifar_100', 'rb'))
        # params, shards, psi, cells = pickle.load(open('cifar_100_klisa_10bp.obj', 'rb'))
        open('klisa_cifar_100.txt','w')
        dimension = 6
        length = int(lst.shape[0]/(2**12))
        pdb.set_trace()
        mins, maxs, data = kd.get_leaves(lst, length)
        
        pdb.set_trace()

        bounding_rects = []
        indices = {}
        cells = []
    
        for i in range(len(mins)):
            rect = (tuple(mins[i]), (tuple(maxs[i])))
            bounding_rects.append(rect)
            indices[rect] = i
    
        # Sort based off of monotonicity condition.
        pdb.set_trace()
        bounding_rects_sorted = bounding_rects.copy()
        mergeSort(bounding_rects_sorted)
        k = np.asarray([indices[i] for i in bounding_rects_sorted])
    
        mins = mins[k]
        maxs = maxs[k]
        data = [data[i] for i in k]
        
        # Create the cells
        # TODO: bounding rectangles for the top most cells must be changed by about .1.
        for i in range(len(mins)):
            cell = Cell(i, (mins[i], maxs[i]))
            for j in data[i]:
                cell.add_data(j)
            cells.append(cell)
            print(f'Cell: {i}, data: {len(cell.data)}')

        pdb.set_trace()
    
        # Check cells:
        for i in range(len(mins)-1, -1, -1):
            br1 = cells[i].bounding_rectangle
            for j in range(i-1,-1, -1):
                br2 = cells[j].bounding_rectangle
                if (br1[0] < br2[1]).all():
                    print(i, j)
                    # raise Exception(f'cell {i} is greater than cell {j}')

        partitions, full_lst = mapping_list_partition_cells(cells, 10)
        pdb.set_trace()
        psi = 50
        params = train(partitions, 10, psi) # Train the partitions with 2 breakpoints. Tunable hyperparameter. IE sigma + 1 == second parameter.
        shards = create_shards(params, full_lst, psi, cells)
        pickle.dump([params, shards, psi, cells], open("cifar_100_klisa_10bp.obj", 'wb'))

        q_points = pickle.load(open('qpoints_images.dump', 'rb'))
        for K in [1, 5, 10, 50, 100, 500]:
            pages = 0
            for point in q_points:
                nearest, p = KNN_updated(K, point, params, shards, psi, cells)
                pages += p
                actual = np.asarray([np.asarray((distance(point, i), *i)) for i in lst])
                actual = actual[np.argsort(actual[:,0])][:K]
                print(f"Point: {point}")
                print(f"Neighbours: {np.asarray(nearest)}")
                print(f"Actual: {actual}")
                print(f"Pages: {p}")
                if not (actual[:, 1:] == np.asarray(nearest)[:, 1:]).all():
                    print(actual)
                    print(nearest)
                    out.write("BIG OOF at {amount} {K}")

            with open("klisa_cifar_10.txt", "a") as out:
                out.write(f"Average pages for {K}: {pages/100}.\n")
            print(f"Average pages: {pages/100}")

    def test_synthetics():
        open("klisa_synthetic_time.txt", 'w')
        for amount in [1000, 4000, 8000, 16000, 32000, 64000]:
            lst = pickle.load(open(f'synthetic_{amount}.dump', 'rb'))

            dimension = 2

            # TODO: http://sid.cps.unizar.es/projects/ProbabilisticQueries/datasets/ Long beach dataset.
            mins, maxs, data = kd.get_leaves(lst, 16) # Should have a total of 32 leaves
            pdb.set_trace()
    
            bounding_rects = []
            indices = {}
            cells = []
    
            for i in range(len(mins)):
                rect = (tuple(mins[i]), (tuple(maxs[i])))
                bounding_rects.append(rect)
                indices[rect] = i
    
            # Sort based off of monotonicity condition.
            pdb.set_trace()
            bounding_rects_sorted = bounding_rects.copy()
            mergeSort(bounding_rects_sorted)
            k = np.asarray([indices[i] for i in bounding_rects_sorted])
    
            mins = mins[k]
            maxs = maxs[k]
            data = [data[i] for i in k]
            
            # Create the cells
            for i in range(len(mins)):
                maxs[i] = maxs[i] + 0.01
                cell = Cell(i, (mins[i], maxs[i]))
                for j in data[i]:
                    cell.add_data(j)
                cells.append(cell)
                print(f'Cell: {i}, data: {len(cell.data)}')
    
            # Check cells:
            for i in range(len(mins)-1, -1, -1):
                br1 = cells[i].bounding_rectangle
                for j in range(i-1,-1, -1):
                    br2 = cells[j].bounding_rectangle
                    if (br1[0] < br2[1]).all():
                        print(i, j)
                        # raise Exception(f'cell {i} is greater than cell {j}')
    
            partitions, full_lst = mapping_list_partition_cells(cells, 10)
            psi = 50
            pdb.set_trace()
            a = time.time()
            params = train(partitions, 3, psi) # Train the partitions with 2 breakpoints. Tunable hyperparameter. IE sigma + 1 == second parameter.
            a = time.time() - a
    
            shards = create_shards(params, full_lst, psi, cells)
        
            q_rects = pickle.load(open(f"synthetic_qrects_{amount}.dump", "rb"))
    
            total_p = 0
            count = 0
            f = open("klisa_synthetic_time.txt", "a")
            f.write(f"Took {a} time training.\n")
            f.close()
    
            KNN_random_points = pickle.load(open(f"synthetic_qpoints_{amount}.dump", "rb"))
            timer = []
            
            for K in [50]:
            # for K in [1, 5, 10, 50, 100, 500]:
                pages = 0
                print(f'K: {pages}')
                for point in KNN_random_points:
                    a = time.time()
                    nearest, p = KNN_updated(K, point, params, shards, psi, cells)
                    timer.append(time.time() - a)
                    nearest = np.asarray(nearest)
                    nearest = nearest[np.argsort(nearest[:, 0])]
                    pages += p
                    actual = np.asarray([np.asarray((distance(point, i), i[0], i[1])) for i in lst])
                    actual = actual[np.argsort(actual[:,0])][:K]
                    if not (actual[:, 1:] == np.asarray(nearest)[:, 1:]).all():
                        print(actual)
                        print(nearest)
                        raise Exception(amount, K)
                    # print(actual[:,1:] == np.asarray(nearest)[:,1:])
                    print(f"Point: {point}")
                    print(f"Neighbours: {np.asarray(nearest)}")
                    print(f"Pages: {p}")
                with open("klisa_synthetic_time.txt", 'a') as output:
                    output.write(f"Average pages for {K} on synthetic points {amount}: {pages/100}.\nAverage time: {sum(timer)/100}\n")
                print(f"Average pages for {K}: {pages/100}")

    def test_long_beach():
        lst = pickle.load(open('LB.dump', 'rb'))
        # params, shards, psi, cells = pickle.load(open("long_beach_klisa_10bp.obj", 'rb'))
        open("klisa_long_beach_query.txt", 'w')

        dimension = 2

        # TODO: http://sid.cps.unizar.es/projects/ProbabilisticQueries/datasets/ Long beach dataset.
        mins, maxs, data = kd.get_leaves(lst, 16) # Should have a total of 32 leaves
        pdb.set_trace()
    
        bounding_rects = []
        indices = {}
        cells = []
    
        for i in range(len(mins)):
            rect = (tuple(mins[i]), (tuple(maxs[i])))
            bounding_rects.append(rect)
            indices[rect] = i
    
        # Sort based off of monotonicity condition.
        pdb.set_trace()
        bounding_rects_sorted = bounding_rects.copy()
        mergeSort(bounding_rects_sorted)
        k = np.asarray([indices[i] for i in bounding_rects_sorted])
    
        mins = mins[k]
        maxs = maxs[k]
        data = [data[i] for i in k]
        
        # Create the cells
        for i in range(len(mins)):
            pdb.set_trace()
            maxs[i] = maxs[i] + 0.01
            cell = Cell(i, (mins[i], maxs[i]))
            for j in data[i]:
                cell.add_data(j)
            cells.append(cell)
            print(f'Cell: {i}, data: {len(cell.data)}')
    
        # Check cells:
        for i in range(len(mins)-1, -1, -1):
            br1 = cells[i].bounding_rectangle
            for j in range(i-1,-1, -1):
                br2 = cells[j].bounding_rectangle
                if (br1[0] < br2[1]).all():
                    print(i, j)
                    # raise Exception(f'cell {i} is greater than cell {j}')
    
        partitions, full_lst = mapping_list_partition_cells(cells, 3)
        psi = 50
        pdb.set_trace()
        params = train(partitions, 10, psi) # Train the partitions with 2 breakpoints. Tunable hyperparameter. IE sigma + 1 == second parameter.
        shards = create_shards(params, full_lst, psi, cells)
        pickle.dump([params, shards, psi, cells], open("long_beach_klisa_10bp.obj", 'wb'))


        KNN_random_points = pickle.load(open('qpoints_LB.dump', 'rb'))
        f = open("klisa_long_beach_results.txt", "w")
        f.close()
        for K in [1, 5, 10, 50, 100, 500]:
            pages = 0
            for point in KNN_random_points:
                nearest, p = KNN_updated(K, point, params, shards, psi, cells)
                pages += p
                actual = np.asarray([np.asarray((distance(point, i), i[0], i[1])) for i in lst])
                actual = actual[np.argsort(actual[:,0])][:K]
                print(f"Point: {point}")
                print(f"Neighbours: {np.asarray(nearest)}")
                print(f"Actual: {actual}")
                print(f"Pages: {p}")

            with open("klisa_long_beach_results.txt", "a") as out:
                out.write(f"Average pages for {K}: {pages/100}.\n")
                if not (actual[:, 1:] == np.asarray(nearest)[:, 1:]).all():
                    print(actual)
                    print(nearest)
                    out.write("BIG OOF at {amount} {K}")
            print(f"Average pages: {pages/100}")

    def test_nd():
        open('klisa_synthetic_nd.txt', 'w')
        values = {3: [10, 10, 10], 4:[ 8, 8, 8, 8], 5: [7, 7, 7, 7, 7], 6: [6, 6, 6, 6, 6, 6]}
        # for dimension in [6]:
        for dimension in [3, 4, 5, 6]:
            # for amount in [32000]:
            for amount in [1000, 4000, 8000, 16000, 32000, 64000]:
                lst = pickle.load(open(f'synthetic_{amount}_{dimension}d.dump', 'rb'))
                T_i = values[dimension]
                Theta = create_cells(lst, T_i)
                partitions, full_lst = mapping_list_partition(lst, Theta, T_i, 10)
    
                M = [partitions[0][0]]
                for i in partitions:
                    M.append(i.max())
                psi = 50
                a = time.time()
                params = train(partitions, 3, psi)
                a = time.time() - a
    
                bounding_rectangles = decompose_query(Theta, Theta[:, 0], Theta[:, -1])
                cells = []
                for i in range(len(bounding_rectangles)):
                    cells.append(Cell(i, bounding_rectangles[i]))
    
                shards = create_shards(params, full_lst, psi, cells)
                mins = []
                maxs = []
                for i in range(dimension):
                    mins.append(np.min(lst[:,i]))
                    maxs.append(np.max(lst[:,i]))
                with open("lisa_synthetic_nd.txt", 'a') as output:
                    output.write(f"Dimension {dimension}\nTook {a} time to train.\n")
    
                q_points = pickle.load(open(f'synthetic_qpoints_{dimension}d.dump', 'rb'))
                for K in [500]:
                # for K in [1, 5, 10, 50, 100, 500]:
                    pages = 0
                    # for point in [q_points[0]]:
                    for point in q_points:
                        a = time.time()
                        nearest, p = KNN_updated(K, point, params, shards, psi, cells)
                        print(f'Took {time.time() - a} seconds to KNN')
                        nearest = np.asarray(nearest)
                        nearest = nearest[np.argsort(nearest[:, 0])]
                        pages += p
                        actual = np.asarray([np.asarray((distance(point, i), *i)) for i in lst])
                        actual = actual[np.argsort(actual[:,0])][:K]
                        
                        if not (actual[:, 1:] == np.asarray(nearest)[:, 1:]).all():
                            print(actual)
                            print(nearest)
                            raise Exception(amount, K)
                        print(f"Point: {point}")
                        print(f"Neighbours: {np.asarray(nearest)}")
                        print(f"Pages: {p}")
                    with open("lisa_synthetic_nd.txt", 'a') as output:
                        output.write(f"Average pages for {K} on synthetic points {amount} for dimension {dimension}D: {pages/100}.\n")
                    
                    print(f"Average pages for {K}: {pages/100}")
    def test(in_file, point_file, out_file, dump_file):
        lst = pickle.load(open(in_file, 'rb'))
        # params, shards, psi, cells = pickle.load(open('cifar_100_klisa_10bp.obj', 'rb'))
        open(out_file,'w')
        dimension = 6
        length = int(lst.shape[0]/(2**12))
        pdb.set_trace()
        mins, maxs, data = kd.get_leaves(lst, length)
        
        pdb.set_trace()

        bounding_rects = []
        indices = {}
        cells = []
    
        for i in range(len(mins)):
            rect = (tuple(mins[i]), (tuple(maxs[i])))
            bounding_rects.append(rect)
            indices[rect] = i
    
        # Sort based off of monotonicity condition.
        pdb.set_trace()
        bounding_rects_sorted = bounding_rects.copy()
        mergeSort(bounding_rects_sorted)
        k = np.asarray([indices[i] for i in bounding_rects_sorted])
    
        mins = mins[k]
        maxs = maxs[k]
        data = [data[i] for i in k]
        
        # Create the cells
        # TODO: bounding rectangles for the top most cells must be changed by about .1.
        for i in range(len(mins)):
            cell = Cell(i, (mins[i], maxs[i]))
            for j in data[i]:
                cell.add_data(j)
            cells.append(cell)
            print(f'Cell: {i}, data: {len(cell.data)}')

        pdb.set_trace()
    
        # Check cells:
        for i in range(len(mins)-1, -1, -1):
            br1 = cells[i].bounding_rectangle
            for j in range(i-1,-1, -1):
                br2 = cells[j].bounding_rectangle
                if (br1[0] < br2[1]).all():
                    print(i, j)
                    # raise Exception(f'cell {i} is greater than cell {j}')

        partitions, full_lst = mapping_list_partition_cells(cells, 10)
        pdb.set_trace()
        psi = 50
        a = time.time()
        params = train(partitions, 10, psi) # Train the partitions with 2 breakpoints. Tunable hyperparameter. IE sigma + 1 == second parameter.
        with open(out_file, 'a') as out:
            out.write(f'Took {time.time() - a} to train.\n')
        shards = create_shards(params, full_lst, psi, cells)
        pickle.dump([params, shards, psi, cells], open(dump_file, 'wb'))

        q_points = pickle.load(open(point_file, 'rb'))
        for K in [1, 5, 10, 50, 100, 500]:
            pages = 0
            for point in q_points:
                nearest, p = KNN_updated(K, point, params, shards, psi, cells)
                pages += p
                actual = np.asarray([np.asarray((distance(point, i), *i)) for i in lst])
                actual = actual[np.argsort(actual[:,0])][:K]
                print(f"Point: {point}")
                print(f"Neighbours: {np.asarray(nearest)}")
                print(f"Actual: {actual}")
                print(f"Pages: {p}")
                if not (actual[:, 1:] == np.asarray(nearest)[:, 1:]).all():
                    print(actual)
                    print(nearest)
                    out.write("BIG OOF at {amount} {K}")

            with open(out_file, "a") as out:
                out.write(f"Average pages for {K}: {pages/100}.\n")
            print(f"Average pages: {pages/100}")
if __name__ == "__main__":
    # kd_tree_implementation.test_synthetics()
    
    opts, args = getopt.getopt(sys.argv[1:], 'ki:q:o:d:')
    kd_use = False
    input_file = None
    query_file = None
    output_file = None
    dump_file = None
    pdb.set_trace()
    for opt, arg in opts:
        if opt == '-k':
            kd_use = True
        elif opt == '-i':
            input_file = arg
        elif opt == '-q':
            query_file = arg
        elif opt == '-o':
            output_file = arg
        elif opt == '-d':
            dump_file = arg
    
    pdb.set_trace()
    if kd_use:
        kd_tree_implementation.test(input_file, query_file, output_file, dump_file)
    else:
        test(input_file, query_file, output_file, dump_file)




    # test_images()
    # pdb.set_trace()
    # test('6d_VISUAL', 'qpoints_images.dump', 'lisa_images_VISUAL_results.txt', 'lisa_VISUAL_dump')
    # kd_tree_implementation.test('6d_VISUAL', 'qpoints_images.dump', 'klisa_images_VISUAL_results.txt', 'klisa_VISUAL_dump')
    # kd_tree_implementation.test('6d_cifar_10', 'qpoints_images.dump', 'klisa_images_cifar_10_results.txt', 'klisa_cifar_10_dump')
    # kd_tree_implementation.test('6d_cifar_100', 'qpoints_images.dump', 'klisa_images_cifar_100_results.txt', 'klisa_cifar_100_dump')
    # test_synthetics()
    # test_nd()
    # test_3d()
    # test_long_beach()
    # pdb.set_trace()
    # TODO: http://sid.cps.unizar.es/projects/ProbabilisticQueries/datasets/ Long beach dataset.

