import numpy as np
from data_generation import generate_numbers
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import pdb
"""
Quick Reminders:
Theta = Border points generated along every axis
T_i = length of the border points along each axis
lst = list of points to be trained on.
k = Number of partitions for the mapped values. Essentially equivalent to how many F we will have.
sigma = Set of breakpoints for beta
V = Number of points in a particular partition.
psi = estimated number of keys in a shard. Approximately equal to how many keys can fit into a page.
"""
def check_alpha(alpha):
    """
    Checks if alpha satisfies the fact that we should be continuously increasing.
    """
    alpha = alpha.flatten()[1:]
    for i in range(len(alpha)):
        inds = [True]*(i + 1) + [False]*(len(alpha)-(i + 1))
        if np.sum(alpha, where=inds) < 0:
            return False
    return True

def calculate_loss(A, alpha, y):
    """
    Calculate the loss function from A, alpha, y
    """
    return np.sum(np.apply_along_axis(lambda x: x * x, 1, (A @ alpha) - y))

def train(partitions, sigma, psi):
    """
    Return the parameters for the piecewise monotone linear regression model.

    Partitions are a K x V array. Note that V might not be the same across all K. IE V is partition specific.
    sigma is the number of breakpoints we want. In accordance to the paper, we pass it in as sigma + 1.
    """
    # Change psi as we increase the number of points it can take.
    settings = []

    # TODO: Make it so that we have more than 5 points in a partition. See if beta can descend when we have more to work with.

    for partition in partitions:
        pdb.set_trace()
        V = len(partition)
        y = np.arange(V).reshape([V, 1])
        beta = np.zeros([1,sigma])
        for i in range(sigma):
            ind = np.floor(i * V/sigma)
            # ind = np.floor(i * V/psi)
            beta[0][i] = partition[int(ind)] # Initialize with an integer

        try:
            alpha, A = calculate_alpha(partition, beta, y)
        except:
            # pdb.set_trace()
            # alpha, A = calculate_alpha(partition, beta,y )
            pass
        curr_loss = calculate_loss(A, alpha, y)
        orig_loss = curr_loss
        # print("loss")
        # print(loss)

        num_iterations = 0
        curr_beta_lr = None
        converge = False
        while(num_iterations < 1000):
            print(curr_loss)
            print(beta)
            grad = calculate_gradient(alpha, beta, partition, A, y)
            print(num_iterations)
            print(grad)
            # pdb.set_trace()
            grad = grad.flatten()[1:] # Grab everything except the first element.
            for i in [0.001, 0.01, 0.05, 0.1]: # Possible learning rates
                beta_lr = (grad * i) + beta # Descend in the gradient. TODO: MAKE SURE THIS IS DOING ELEMENT WISE ADDITION
                beta_lr = np.sort(beta_lr)
                # pdb.set_trace()
                try:
                    alpha, A = calculate_alpha(partition, beta_lr, y)
                except:
                    # If we fail to calculate_alpha properly, then we just say, okay, this is considered optimal
                    # pdb.set_trace()
                    continue
                if check_alpha(alpha):
                    loss = calculate_loss(A, alpha, y)
                    if loss < curr_loss:
                        curr_loss = loss
                        curr_beta_lr = beta_lr
            if curr_beta_lr is not None:
                # We found something that had loss minimizing.
                beta = curr_beta_lr
                num_iterations += 1
                if orig_loss - curr_loss < 0.000001:
                    # pdb.set_trace()
                    break
                orig_loss = curr_loss
            else:
                # Loss didn't change, so break?
                break
        alpha, A = calculate_alpha(partition, beta, y)
        settings.append((alpha.flatten(), beta.flatten(), (A @ alpha).flatten()))
    return settings

def calculate_gradient(alpha, beta, partition, A, y):
    """
    Calculates the gradient based on alpha/beta/partition/A/y

    alpha: A (1 x sigma + 2) array. Consists of a bar, a0, a1, a2, ..., a sigma. Therefore sigma + 2
    beta: A (1 x sigma + 1) array. Consists of b0, b1, ..., b_sigma. Therefore sigma + 1
    partition: A V length array representing the things that we do.
    A: A (V + 1 x sigma + 2) array. For calculating alpha
    y: A (V x 1) array to calculate the outputs.

    returns a (1 x sigma + 1) matrix. Should be the same shape as beta passed in.

    """
    K = np.diag(alpha.flatten()) # Construct the diagonal matrix.
    # pdb.set_trace()
    sigma = alpha.shape[0]
    V = len(partition)
    partition = partition.reshape([1,V])
    G = np.zeros([sigma, V])
    G[0] = -1
    count = 1
    for i in beta.flatten():
        G[count] = np.apply_along_axis(lambda x, b: -1 if x >= b else 0, 0, partition, i)
        count += 1
    r = A @ alpha - y
    g = 2 * (K @ G @ r)
    Y = 2 * (K @ G @ G.T @ K.T)
    # pdb.set_trace()
    if np.linalg.det(Y) == 0:
        print('Determinant of 0')
        return -g # Return the gradient if Y is singular
    inv = np.linalg.inv(Y)
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
    y = np.arange(V).reshape([V,1])
    A = np.zeros([V, sigma + 1]) # Remember that a row in A is [1, xi-B0, (xi - B1) if xi >= B1, ..., xi - B_sigma, if x 0 > = B_sigma]. This is len(beta) + 1 == sigma + 1 + 1
    count = 0
    for i in partition:
        row = np.zeros([1, sigma + 1])
        row[0][0] = 1
        row[0][1:] = np.apply_along_axis(lambda b, x: x - b if x > b else 0, 0, beta, i)
        A[count] = row
        count += 1
    inv = np.linalg.inv(np.matmul(A.T, A))
    alpha = np.matmul(np.matmul(inv, A.T), y) # Alpha is a bar, a0, a1 ,..., a_sigma
    return alpha, A



def find_interval(point, Theta, T_i):
    """
    point is a 1 x 2 array.
    Return the interval that we found the point in.

    Will be of form ((x_0, x_1), (y_0, y_1))
    """
    # print(Theta, point)
    x = binary_search(point[0], Theta[0], 0, T_i[0] - 1)
    y = binary_search(point[1], Theta[1], 0, T_i[1] - 1)
    return (x,y)

def binary_search(element, array, beginning, end):
    """
    Return the index of the element that it closest belongs to in an array.
    """
    mid = (beginning + end)//2
    if element < array[beginning]: # End the binary search if the element is less than min
        return None
    if element > array[end]: # In this particular case, we have something belonging in the last interval.
        return array[end], array[end + 1]
    if array[mid] <= element < array[mid + 1]:
        return array[mid], array[mid + 1]
    else:
        if element > array[mid] and element >= array[mid + 1]:
            return binary_search(element, array, mid + 1, end)
        else:
            return binary_search(element, array, beginning, mid)

def mapping_list_partition(lst, Theta, T_i, k):
    """
    Returns a k x F, where F is approximately len(lst)/k equal partition of the list of points.
    Also returns the original point list with an attached mapping to them, sorted by mapping.

    lst is a N x 2 list of points.
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
    Calculates the cell index based on the given point and Theta.
    """
    x,y = find_interval(point, Theta, T_i)
    x_ind = np.where(Theta[0] == x[0])[0][0]
    y_ind = np.where(Theta[1] == y[0])[0][0]

    return x_ind * T_i[1] + y_ind

def mapping_function(point, Theta, T_i):
    """
    Point is a 1 x 2 array.
    Returns the value of the mapping function on point, Theta, T_i
    """
    x,y = find_interval(point, Theta, T_i)
    ind = cell_index(point, Theta, T_i)

    C_i = (y[1] - y[0]) * (x[1] - x[0])
    H_i = (point[0] - x[0]) * (point[1] - y[0])
    # print("area cell {} point {}".format(C_i, H_i))
    # cell_index = np.where(Theta[0] == x[0]) * T_i[0] + np.where(Theta[1] == y[0])
    return ind + (H_i/C_i)

def calculate_border_points(axis, T_i, maximum):
    """
    data is a 1 x N of sorted data along an axis.
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


def create_cells(data, T_i):
    """
    T_i is a N-d array, where T_i[j] is the partition amount for dimension j.
    data is a N x 2 matrix. IE it contains N points in 2d.

    return a Theta that represents all the cells. Consists of:
    Theta[0] = border points for x
    Theta[1] = border points for y
    """
    
    X = np.take(data, 0, 1)
    Y = np.take(data, 1, 1)
    T_ix = T_i[0]
    T_iy = T_i[1]
    maximum = max(T_ix,T_iy)
    
    sort_X = np.sort(X)
    sort_Y = np.sort(Y)

    theta_x = calculate_border_points(sort_X, T_ix, maximum)
    theta_y = calculate_border_points(sort_Y, T_iy, maximum)

    return np.vstack((theta_x, theta_y))

def create_shards(params, full_lst, psi):
    """
    Create the shards based on the params.

    params: List of tuples for index i, representing F_i(x). In other words, contains alpha, beta, and the resulting calculation of F_i(x).
    full_lst: List of points including mapping in the first column.

    Returns a list of shards. shards[i] corresponds to the shard prediction function for learned function F_i
    """
    shards = []
    D = np.ceil(len(params[0][-1])/psi)
    count = 0
    points = full_lst[:, 1:] # Grab everything but the mapping. We want only points and they are already sorted by mapping.
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

def decompose_query(Theta, bottom_left, top_right):
    """
    Takes a query rectangle and we decompose it into cells that intersect with the cells.

    bottom_left: 1x2 numpy array representing the bottom left vertex
    top_right: 1x2 numpy array representing the top right vertex.
    Theta: The breakpoints for all the axis.

    Returns a list of points that overlap with Theta.
    """
    x_1, y_1 = bottom_left.flatten()
    x_2, y_2 = top_right.flatten()
    lst_x = [x_1]
    lst_y = [y_1]
    for i in Theta[0]: # Dimension x
        if x_1 <= i < x_2:
            lst_x.append(i)
    for i in Theta[1]: # Dimension y
        if y_1 <= i < y_2:
            lst_y.append(i)
    lst_x.append(x_2)
    lst_y.append(y_2)
    points_x = []
    points_y = []
    for i in range(len(lst_x) - 1):
        points_x.append(( lst_x[i], lst_x[i+1]))

    for i in range(len(lst_y) - 1):
        points_y.append((lst_y[i], lst_y[i+1]))

    full_points = [tuple(zip(a,b)) for a in points_x for b in points_y]
    return full_points

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

def shard_get(index, lower, upper, shards):
    k = []
    shard = shards[index]
    if lower is None and upper is None:
        # Return everything within the shard at index index.
        for i in shard:
            k.extend(shard[i])
    elif lower is None:
        upper = int(upper)
        for i in range(0, upper + 1):
            k.extend(shard[i])
    elif upper is None:
        lower = int(lower)
        i = lower
        while i in shard:
            k.extend(shard[i])
            i += 1
    else:
        upper, lower = int(upper), int(lower)
        for i in range(lower, upper + 1):
            k.extend(shard[i])
    return k

def find_points(points, params, shards, M, T_i, psi):
    """
    points is a N x 2 list of points that we need to calculate the shards on.
    params is a K X 3 list representing alpha, beta, and F_i in that order
    shards is a list of dictionaries that represents what each shard[i] sets.
    M is a 1 x M lenght list representing the partition points.
    T_i is a 2 x N length list representing the lengths of Theta.
    psi is a integer representing the expected number of keys falling in a range.

    Returns a list of points found that lie within points.
    """
    answer = set()
    for i in points:
        lower_left, upper_right = i
        lower_mapping = mapping_function(lower_left, Theta, T_i)
        upper_mapping = mapping_function(upper_right, Theta, T_i)
        lower = binary_search(lower_mapping, M, 0, len(M) - 2) # We remove 2 from length M, because the bin search will return an interval
        upper = binary_search(upper_mapping, M, 0, len(M) - 2)

        index_lower = M.index(lower[0])
        index_upper = M.index(upper[0])

        shard_pred_l = shard_prediction(lower_mapping, index_lower, params, psi)
        shard_pred_u = shard_prediction(upper_mapping, index_upper, params, psi)
        k = []
        if index_upper < index_lower:
            raise Exception("Upper should be higher than lower, failed")
        if index_upper > index_lower:
            print("We got a problem. Need to rethink how to grab shards...")
            k = shard_get(index_lower, shard_pred_l, None, shards)
            for j in range(index_lower + 1, index_upper):
                k.extend(shard_get(j, None, None, shards))
            k.extend(shard_get(index_upper, None, shard_pred_u, shards))
        else:
            k = shard_get(index_lower, shard_pred_l, shard_pred_u, shards) 
        answer.update([tuple(i) for i in k])
    return answer

def visualize(Theta, lst, T_i):
    x = np.take(lst, 0, 1)
    y = np.take(lst, 1, 1)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for i in range(T_i[0]): # We need to know the length of the T_i
        for j in range(T_i[1]):
            ax.add_patch(Rectangle(xy=(Theta[0][i], Theta[1][j]), width=Theta[0][i+1]-Theta[0][i], height=Theta[1][j+1]-Theta[1][j], fill=False, color='blue'))
    ax.scatter(x, y, color='red')
    # annotations = np.zeroes(lst.shape[0]) # N points, with annotations.
    # i = 0
    # while i < lst.shape[0]:
    #     mapping_function(lst[i], Theta, T_i)
    #     # annotation = mapping_function(lst[i], Theta, T_i)
    #     # ax.annotate(mapping_function(lst[i],Theta, T_i), lst[i])
    #     i += 1

    plt.show()

def in_query(point, q_rectangle):
    low, up = q_rectangle
    x_l, y_l = low
    x_u, y_u = up

    x_p, y_p = point
    return x_l <= x_p <= x_u and y_l <= y_p <= y_u

if __name__ == "__main__":
    lst = generate_numbers(0, 100, 100)
    T_i = [5, 5]
    # print("lst {}".format(lst))
    Theta = create_cells(lst, T_i)
    # print(Theta)
    # print(cell_index(lst[0], Theta, T_i))
    partitions, full_lst = mapping_list_partition(lst, Theta, T_i, 3) # Create 3 equal length partitions of the mapping space.
    M = [partitions[0][0]]
    for i in partitions:
        M.append(i.max())
    params = train(partitions, 3, 5) # Train the partitions with 2 breakpoints. Tunable hyperparameter. IE sigma + 1 == second parameter. Also psi = 5 because why not. Each shard generates ~ 5.
    shards = create_shards(params, full_lst, 5)
    print(shards)
    print(params)
    points = decompose_query(Theta, np.asarray((20, 20)), np.asarray((70, 70)))
    return_results = find_points(points, params, shards, M, T_i, 5)
    print(len(return_results))
    final_results = {point for point in return_results if in_query(point, ((20, 20), (70, 70)))}
    pdb.set_trace()
    another_result = [point for point in full_lst[:, 1:]if in_query(point, ((20, 20), (70, 70)))]
    pdb.set_trace()
    visualize(Theta, lst, T_i )
    
