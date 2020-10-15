import numpy as np
from data_generation import generate_numbers
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

def find_interval(point, Theta, T_i):
    """
    Return the interval that we found the point in.

    Will be of form ((x_0, x_1), (y_0, y_1))
    """
    print(Theta, point)
    x = binary_search(point[0], Theta[0], 0, T_i[0] - 1)
    y = binary_search(point[1], Theta[1], 0, T_i[1] - 1)
    return (x,y)

def binary_search(element, array, beginning, end):
    """
    Return the index of the element that it closest belongs to in an array.
    """
    mid = (beginning + end)//2
    if element > array[end]: # In this particular case, we have something belonging in the last interval.
        return array[end], array[end + 1]
    if array[mid] <= element < array[mid + 1]:
        return array[mid], array[mid + 1]
    else:
        if element > array[mid] and element >= array[mid + 1]:
            return binary_search(element, array, mid + 1, end)
        else:
            return binary_search(element, array, beginning, mid)

def cell_index(point, Theta, T_i):
    """
    Calculates the cell index based on the given point and Theta.
    """
    x,y = find_interval(point, Theta, T_i)
    x_ind = np.where(Theta[0] == x[0])[0][0]
    y_ind = np.where(Theta[1] == y[0])[0][0]
    return x_ind * T_i[0] + y_ind

def mapping_function(point, Theta, T_i):
    """
    Returns the value of the mapping function on point
    """
    x,y = find_interval(point, Theta, T_i)
    ind = cell_index(point, Theta, T_i)
    print("cell {}".format((x,y)))
    print("Point {}".format(point))

    C_i = (y[1] - y[0]) * (x[1] - x[0])
    H_i = (point[0] - x[0]) * (point[1] - y[0])
    print("area cell {} point {}".format(C_i, H_i))
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
    i = 0
    while i < lst.shape[0]:
        mapping_function(lst[i], Theta, T_i)
        # annotation = mapping_function(lst[i], Theta, T_i)
        ax.annotate(mapping_function(lst[i],Theta, T_i), lst[i])
        i += 1

    plt.show()


if __name__ == "__main__":
    lst = generate_numbers(0, 100, 10)
    T_i = [2, 3]
    print("lst {}".format(lst))
    Theta = create_cells(lst, T_i)
    print(Theta)
    print(cell_index(lst[0], Theta, T_i))
    visualize(Theta, lst, T_i )
    
