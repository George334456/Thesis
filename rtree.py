from typing import Tuple, List, Optional
import numpy as np
from data_generation import cluster, generate_numbers
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.collections import PolyCollection
from matplotlib.patches import Rectangle
import pdb
import math
import pickle
import operator
from functools import reduce
import time

M =50 # Arbitrarily set M.
# M = 2
def distance(point1, point2, dimension):
    """
    Calculate the distance between 2 points
    point1: 1xdimension array representing first point
    point2: 1xdimension array representing second point

    Returns the square distance between the two.
    """
    answer = 0
    for i in range(dimension):
        answer += (point1[i]-point2[i]) ** 2
    
    return answer


def prune(branches, nearest, k):
    """
    Prunes a set of branches based on MINDIST and MINMAXDIST of branches, as well as based off of nearest.
    """
    cont = True
    max_nearest = max([i[0] for i in nearest]) if len(nearest) == k else float('inf')
    while cont:
        if len(nearest) < k:
            # We don't want to prematurely prune branches. Ex: We could prune all but one branch, but this branch contains x < k elements. KNN will fail.
            break
        lst = [(i[0], i[1]) for i in branches]
        prune_set = set()
        for i, e in enumerate(lst):
            curr_min_max_dist = e[0]
            # if i in prune_set:
            #     continue
            if e[1] > max_nearest:
                prune_set.add(i)

        if len(prune_set) == 0:
            # Finished pruning.
            break
        else:
            # Something to prune, need to prune and rerun
            branches = [e for i, e in enumerate(branches) if i not in prune_set]
    return branches

def bounding_rectangle(points):
    """
    Points is a np array of N points of dimension elements. In other words, this is a (N, dimension) np array.

    Returns the bounding box that encloses all points in points. This is in format [(lower_left_0, ..., lower_left_(dimension - 1)), (top_right_0, ..., top_right_(dimension-1))]
    """
    points = np.asarray(points)
    dimension = points.shape[1]
    answer = np.zeros((2, dimension))
    for i in range(dimension):
        curr = np.take(points, i, 1)
        answer[0][i] = np.amin(curr)
        answer[1][i] = np.amax(curr)

    return answer

def min_max_dist(point, rectangle):
    """
    point is a 1x dimension np array. Represents a point
    rectangle is a 2x dimension np array. Represents the bottom-left and top-right corners respectively

    Returns the "MINMAXDIST" as required by the paper.
    """

    rect_points = np.stack((np.take(rectangle, 0, 1), np.take(rectangle, 1, 1)))

    dimension = rectangle.shape[1]
    S = 0
    for i in range(dimension):
        interval = rectangle[0][i], rectangle[1][i]
        rM = interval[0] if point[i] >= (interval[0] + interval[1])/2 else interval[1]
        S += (point[i] - rM) ** 2
    
    minimum = float('inf')
    for i in range(dimension):
        interval = rectangle[0][i], rectangle[1][i]
        rM = interval[0] if point[i] >= (interval[0] + interval[1])/2 else interval[1]
        rm = interval[1] if point[i] >= (interval[0] + interval[1])/2 else interval[0]
        minimum = min(minimum, S - (point[i] - rM) ** 2 + (point[i] - rm) ** 2)
    return minimum

def min_dist(point, rectangle):
    """
    point is a 1xdimension np array. Represents a point
    rectangle is a 2xdimension np array. Represents the bottom-left and top-right corners respectively

    Returns the "minimum_distance" as required by the paper.
    """
    dimension = rectangle.shape[1]
    count = 0

    for i in range(dimension):
        mini = rectangle[0][i]
        maxi = rectangle[1][i]
        r = 0
        if point[i] < mini:
            r = mini
        elif point[i] > maxi:
            r = maxi
        else:
            r = point[i]
        count += (point[i] - r) ** 2

    return count

def adjust_rectangle(node, dimension):
    """
    Given an RTree_node, return the bounding rectangle that contains all its children rectangles.
    """
    rectangles = np.empty([1,dimension])
    for i in node.children:
        rectangles = np.concatenate((rectangles, i.rectangle))
    
    rectangles = np.delete(rectangles, 0 ,0)
    return bounding_rectangle(rectangles)

def overlap(point, rect):
    """
    Find if point overlaps rectangle rect.

    point is a 1xdimension array representing a point.
    rect is a 2xdimension array representing first the minimum of all points (bottom_left corner equivalent in 2d) and the maximum of all points (top_right corner equivalent in 2d)
    """
    dimension = rect.shape[1]
    for i in range(dimension):
        k = rect[0][i] <= point[i] <= rect[1][i]
        if not k:
            return False
    return True

def overlap_1d(line1, line2):
    """
    Checks if two lines overlap.

    line1: 1x2 array representing the beginning and end of a line
    line2: 1x2 array representing the beginning and end of a line

    Returns if the two of them overlap
    """
    min1, max1 = line1
    min2, max2 = line2

    return min1 <= min2 <= max1 or min1 <= max2 <= max1 or min2 <= max1 <= max2 or min2 <= min1 <= max2

def overlap_rectangles(rect1, rect2):
    """
    Determine if two rectangles overlap.

    rect1 and rect2 are 2 x 2 arrays.
    """
    # https://stackoverflow.com/questions/20925818/algorithm-to-check-if-two-boxes-overlap
    # TODO: FOR HIGHER DIMENSIONS
    mins1, maxs1 = rect1
    mins2, maxs2 = rect2

    dimension = rect1.shape[1]
    is_overlap = False
    for i in range(dimension):
        is_overlap = overlap_1d(rect1[:, i], rect2[:, i])
        if not is_overlap:
            return False
    return True

class RTree_node:
    """
    This class represents the internal nodes of an R-tree
    The children are going to be addresses of a lower node. The rectangle will cover the entire points encapsulated by the children.
    """
    ID = 0
    def __init__(self, rectangle: List, children, parent, data_points = []):
        """Sets the stuff for a node

        If a node has children, then it is not a leaf, and the data_point position has no meaning.
        If a node doesn't have children, then it is considered a leaf, and the data_point data can be accessed
        Parent is the parent Node. For the root, Parent is None
        """
        self.parent = parent
        self.rectangle = rectangle
        self.children = children
        self.data_points = data_points
        self.id = RTree_node.ID
        RTree_node.ID += 1
    
    def insert_to_leaf(self, point):
        """
        Inserts a point into the node

        Point is a (2,) np array.

        Returns a tuple of nodes n1 and n2.
        If n2 is filled, a split has happened.
        If n2 is None, a split has not been made, and n1 has simply been added to.
        """
        self.data_points.append(point)
        n1 = None
        n2 = None
        if len(self.data_points) > M:
            n1, n2 = self.split_node()
        else:
            self.rectangle = bounding_rectangle(self.data_points)
            n1 = self

        return n1, n2
    
    def insert_to_node(self, node):
        self.children.append(node)
        n1 = None
        n2 = None
        if len(self.children) > M:
            n1, n2 = self.split_node()
        else:
            n1 = self
        return n1, n2

        # Update bounding rectangle.

    def split_node(self):
        if self.children == []:
            # This is a leaf node
            lst = []
            for i in self.data_points:
                lst.append(i)
            arr = np.array(lst)
            lst1, lst2, kmeans = cluster(arr, M)
            # Node 1 modifies itself (to avoid looking it up)
            self.rectangle = bounding_rectangle(lst1)
            self.children = []
            self.data_points = list(lst1)
            node1 = self
            node2 = RTree_node(bounding_rectangle(lst2), [], self.parent, list(lst2))
            return node1, node2
            #node1 = RTreeNode(self.parent, b_rectangle_1, [], lst1)
            #node2 = RTreeNode(self.parent, brectangle_2, [], lst2)
            #return (node1, node2)

        else:
            lst = []
            mapping = {}
            # Use a mapping of centroid point to index, since we get a list of points back
            # Use this mapping to construct a children list to update.
            for i, e in enumerate(self.children):
                rect = e.rectangle
                dimension = rect.shape[1]
                centroid = []
                for j in range(dimension):
                    curr_min = rect[0][j]
                    curr_max = rect[1][j]
                    centroid.append((curr_max + curr_min)/2)
                centroid = tuple(centroid)
                lst.append(centroid)
                mapping[centroid] = i

            arr = np.array(lst)
            lst1, lst2, kmeans = cluster(arr, M)
            node2 = RTree_node([], [self.children[mapping[tuple(i)]] for i in lst2], self.parent, [])
            for child in node2.children:
                child.parent = node2

            self.children = [self.children[mapping[tuple(i)]] for i in lst1]
            self.data_points = []
            node1 = self
            return node1, node2
    
    def search(self, rectangle):
        """
        Search for all points within a query rectangle
        
        Returns a list of all points found within the query box.
        """
        # Start at root
        ans = []
        pages = 1 # Have to read self once.
        ids_found = []

        if self.children == []:
            # We are a leaf node.
            for i in self.data_points:
                if overlap(i, rectangle):
                    ans.append(i)
                    ids_found.append(self.id)
        else:
            # Loop through children
            # Let's use DFS for this.
            for child in self.children:
                rect = child.rectangle
                if overlap_rectangles(rect, rectangle):
                    a, p, new_ids = child.search(rectangle)
                    ans.extend(a)
                    ids_found.extend(new_ids)
                    pages += p
        return ans, pages, ids_found

    def parent_chain(self):
        chain = [self.id]
        curr = self.parent
        while curr is not None:
            chain.append(curr.id)
            curr = curr.parent
        return chain

    def get_by_id(self, id):
        if self.id == id:
            return self
        else:
            for i in self.children:
                k = i.get_by_id(id)
                if k:
                    return k
            return False


    def KNN(self, point, nearest, k):
        """
        point: 1x dimension array representing the point
        nearest: List representing the nearest k positions so far.
        k: The number of nearest neighbours to find.
        """
        dimension = len(point)
        pages = 0
        if self.children == []:
            # We are a leaf
            for i in self.data_points:
                dist = distance(i, point, dimension)
                dist_entry = np.insert(i, 0, dist)

                # dist_entry is supposed to be of form [distance, point 0, point 1, ... point n]
                if len(nearest) < k:
                    nearest.append(dist_entry)
                else:
                    # Sort the objects found in the children.
                    nearest = np.asarray(nearest)
                    nearest = np.concatenate((nearest,dist_entry.reshape(1,dimension + 1)), 0)
                    nearest = nearest[np.argsort(nearest[:,0])][:k]
                    nearest = list(nearest)
            pages += 1
        else:
            # We aren't a leaf.
            branches = []
            for i in self.children:
                branches.append((min_max_dist(point, i.rectangle), min_dist(point, i.rectangle), i))
                # sort by min_max_dist
            branches = np.asarray(branches)
            branches = branches[np.argsort(branches[:,0])]
            branches = list(branches)
            branches = prune(branches, nearest, k)

            while len(branches) > 0:
                child = branches.pop(0)[-1]
                nearest, p = child.KNN(point, nearest, k)
                pages += p
                branches = prune(branches, nearest, k)

        return nearest, pages

        
class RTree:
    """
    Implementation of an R-Tree. It only stores the root.
    """
    def __init__(self, dimension):
        """
        Initialize a root node with no bounding rectangle, no children, no parent, and no data points.
        """
        RTree_node.ID = 0
        self.root = RTree_node([], [], None, [])
        self.data = []
        self.dimension = dimension


    def adjust_tree(self, n1, n2):
        # if n2 is None, then no split has happened.
        # If n2 has a value, then a split has happened.
        
        # First, n1 has always been modified. We have to modify the parent bounding boxes.
        # Then, if n2 has been created, we have to insert into nodes, splitting if necessary.
        # We also have to do a root check.
        p1 = n1.parent
        p2 = None
        if p1 is None:
            # We are at root. Stop
            return n1, n2
        else:
            if n2: # A split has occurred, we need to insert into parent
                p1, p2 = p1.insert_to_node(n2)
            # Adjust the bounding rectangle on parent, by looking at the children
            if p2:
                p2.rectangle = adjust_rectangle(p2, self.dimension)
            p1.rectangle = adjust_rectangle(p1, self.dimension)
           
            return self.adjust_tree(p1, p2)
    
    def chooseLeaf(self, start: RTree_node, point: Tuple[float, float]):
        """
        Choose the leaf on which to insert this point.

        Takes a starting node, which is an RTree node, and a point.
        """
        curr = start
        if curr.children == []:
            return curr
        else:
            min_increase = float("inf")
            min_node = None
            for i in curr.children:
                rect = i.rectangle
                dimension = rect.shape[1]
                box_lengths = []
                point_lengths = []
                curr_area = 0
                # Calculate lengths across each dimension
                for j in range(dimension):
                    box_lengths.append(rect[1][j] - rect[0][j])
                    point_lengths.append(max(point[j], rect[1][j]) - min(point[j], rect[0][j]))

                curr_area = reduce(operator.mul, box_lengths, 1)
                new_area = reduce(operator.mul, point_lengths, 1)

                if new_area - curr_area < min_increase:
                    min_node = i
                    min_increase = new_area - curr_area
                # TODO: Check if we're choosing the right subtree.
            return self.chooseLeaf(min_node, point)

    def insert(self, point: Tuple[float, float]):
        self.data.append(point)
        leaf = self.chooseLeaf(self.root, point)
        n1, n2 = leaf.insert_to_leaf(np.array(point))
        n1, n2 = self.adjust_tree(n1, n2)
        if n2 and n1.parent == None:
            # We had to split at root.
            self.root = RTree_node(bounding_rectangle(self.data), [n1, n2], None, []) 
            n1.parent = self.root
            n2.parent = self.root

def data_len(root):
    if root.children == []:
        return len(root.data_points)
    else:
        length = 0
        for i in root.children:
            length += data_len(i)
        return length

def traverse3d(root, ax):
    for i in root.children:
        rect = i.rectangle
        mins = rect[0]
        maxs = rect[1]
        faces = []

        # https://stackoverflow.com/questions/44881885/python-draw-parallelepiped/49766400#49766400
        for i in range(3):
            curr_face = []
            curr_face_max = []
            curr_face.append(mins)
            curr_face_max.append(maxs)
            current_dimensions = [j for j in list(range(3)) if j != i]
            
            # visit dimension 0
            dim0 = current_dimensions[0]
            length = maxs[dim0] - mins[dim0]

            k = mins.copy()
            k[dim0] = mins[dim0] + length
            curr_face.append(k)

            k = maxs.copy()
            k[dim0] = maxs[dim0] - length
            curr_face_max.append(k)

            # Visit dimension 0 and 1
            dim1 = current_dimensions[1]
            length2 = maxs[dim1] - mins[dim1]
            
            k = mins.copy()
            k[dim1] = mins[dim1] + length2
            k[dim0] = mins[dim0] + length
            curr_face.append(k)

            k = maxs.copy()
            k[dim1] = maxs[dim1] - length2
            k[dim0] = maxs[dim0] - length

            curr_face_max.append(k)

            # visit dimension 1
            k = mins.copy()
            k[dim1] = mins[dim1] + length2
            curr_face.append(k)

            k = maxs.copy()
            k[dim1] = maxs[dim1] - length2
            curr_face_max.append(k)
            faces.append(curr_face)
            faces.append(curr_face_max)
        
        for k in faces:
            print(k)
        ax.add_collection3d(Poly3DCollection(faces, facecolors='cyan', linewidths=1, edgecolors='r', alpha=.25))


def traverse(root, fig):
    if root.children == []:
        print(len(root.data_points))
    count = 0
    for i in root.children:
        rect = i.rectangle
        min_x = rect[0][0]
        min_y = rect[0][1]
        max_x = rect[1][0]
        max_y = rect[1][1]
        print("adding_patch")
        fig.annotate(count, (max_x, max_y))
        fig.add_patch(Rectangle(xy=(min_x, min_y), width=max_x-min_x, height=max_y-min_y, fill=False, color='blue'))
        # traverse(i, ax)
        count += 1

def test_images():
    root = RTree(6)
    lst = pickle.load(open('6d_cifar_100', 'rb'))
    # root = pickle.load(open("LB_rtree.obj", 'rb'))
    for i in lst:
        root.insert(i)
    total = 0
    q_points = pickle.load(open('qpoints_images.dump', 'rb'))

    print("KNN Testing!")
    f = open("rtree_images_cifar_100_results.txt", 'w')
    f.close()
   
    for k in [1,5,10, 50,100, 500]:
        total = 0
        for i in q_points:
            neighbours, pages = root.root.KNN(i, [], k)
            neighbours = np.asarray(neighbours)
            actual = np.asarray([np.asarray((distance(point, i, 2), point[0], point[1])) for point in lst])
            actual = actual[np.argsort(actual[:,0])]
            print(actual[:k])

            total += pages
            print(f'Point {i}')
            print(f'Neighbours {neighbours}')
            print(f'Pages {pages}')
        with open("rtree_images_cifar_100_results.txt", 'a') as out:
            out.write(f'Average pages for {k}: {total/100}\n')
        print(f'Average pages is {total/100}')

def test_long_beach():
    root = RTree(2)
    lst = pickle.load(open('LB.dump', 'rb'))
    # root = pickle.load(open("LB_rtree.obj", 'rb'))
    fig = plt.figure()

    ax = fig.add_subplot(111)
    for i in lst:
        root.insert(i)
    pickle.dump(root, open("LB_rtree.obj", 'wb'))
    x = np.take(lst,0, 1)
    y = np.take(lst, 1, 1)
    ax.scatter(x,y, color = 'red')
    total = 0
    q_rects = pickle.load(open("query_rectangles_long_beach.dump", 'rb'))

    count = 0
    for i in q_rects:
        i = np.asarray(i)
        result, pages, ids = root.root.search(i)
        print(f"Results: {len(result)}")
        # actual = [point for point in lst if overlap(point, i)]
        # print(f"Actual: {len(actual)}")
        total += pages
        print(f'Pages {pages}')
        count += 1
    print(f'Average pages {total/100}')

    result, pages, ids = root.root.search(np.asarray(((20, 20), (45, 45))))
    print(f'pages {pages}')

    q_points = pickle.load(open('qpoints_LB.dump', 'rb'))

    print("KNN Testing!")
    f = open("rtree_long_beach_results.txt", 'w')
    f.close()
   
    for k in [1,5,10, 50,100, 500]:
        total = 0
        for i in q_points:
            neighbours, pages = root.root.KNN(i, [], k)
            neighbours = np.asarray(neighbours)
            actual = np.asarray([np.asarray((distance(point, i, 2), point[0], point[1])) for point in lst])
            actual = actual[np.argsort(actual[:,0])]
            print(actual[:k])

            total += pages
            print(f'Point {i}')
            print(f'Neighbours {neighbours}')
            print(f'Pages {pages}')
        with open("rtree_long_beach_results.txt", 'a') as out:
            out.write(f'Average pages for {k}: {total/100}\n')
        print(f'Average pages is {total/100}')

    traverse(root.root, ax)
    plt.show()

def test_synthetics():
    open("rtree_synthetic_2d.txt", 'w')
    # for amount in [1000]: 
    for amount in [1000, 4000, 8000, 16000, 32000, 64000]:
        lst = pickle.load(open(f'synthetic_{amount}.dump', 'rb'))
        root = RTree(2)
        for i in lst:
            # print(data_len(root.root))
            root.insert(i)

        # root.root.search(np.asarray(((1446, 322), (1447, 323))))
        q_rects = pickle.load(open(f'synthetic_qrects_{amount}.dump', 'rb'))
        total = 0

        count = 0
        for i in q_rects:
            i = np.asarray(i)
            result, pages, ids = root.root.search(i)
            actual = [point for point in lst if overlap(point, i)]
            print(f"Actual: {len(actual)}")
            print(f'results: {len(result)}')
            total += pages
            print(f'Pages {pages}')
            count += 1
        print(f'Average pages {total/100}')

        KNN_random_points = pickle.load(open(f'synthetic_qpoints_{amount}.dump', 'rb'))
        count = 0
        timer = []
        for K in [50]:
        # for K in [1, 5, 10, 50, 100, 500]:
            pages = 0
            print(f'K: {K} Current Pages: {pages}')
            for point in KNN_random_points:
                a = time.time()
                neighbours, p = root.root.KNN(point, [], K)
                timer.append(time.time() - a)
                neighbours = np.asarray(neighbours)
                pages += p
                actual = np.asarray([np.asarray((distance(point, i, 2), i[0], i[1])) for i in lst])
                actual = actual[np.argsort(actual[:,0])][:K]
                if not (actual[:, 1:] == np.asarray(neighbours[:, 1:])).all():
                    pdb.set_trace()
                    print(data_len(root.root))
                    root.root.KNN(point, [], K)
                    print(actual)
                    print(neighbours)
                    raise Exception(amount, count)
                    print(amount)
                # print(actual[:, 1:] == np.asarray(neighbours)[:, 1:])
                print(f"Point: {point}")
                print(f"Neighbours: {neighbours}")
                print(f'Pages {p}')
                count += 1
            with open("rtree_synthetic_2d.txt", 'a') as output:
                output.write(f"Average pages for {K} on synthetic points {amount}: {pages/100}.\nTime is {sum(timer)/100}\n")
            print(f"Average pages for {K}: {pages/100}")


def test_1000():
    root = RTree(2)
    lst = pickle.load(open('data_1000.dump', 'rb'))
    fig = plt.figure()

    ax = fig.add_subplot(111)
    for i in lst:
        root.insert(i)
    x = np.take(lst,0, 1)
    y = np.take(lst, 1, 1)
    ax.scatter(x,y, color = 'red')
    total = 0
    # q_rects = pickle.load(open("query_rectangles_100_100x100.dump", 'rb'))

    # count = 0
    # for i in q_rects:
    #     i = np.asarray(i)
    #     result, pages, ids = root.root.search(i)
    #     print(f"Results: {len(result)}")
    #     total += pages
    #     print(f'Pages {pages}')
    #     count += 1
    # print(f'Average pages {total/100}')

    # result, pages, ids = root.root.search(np.asarray(((20, 20), (45, 45))))
    # print(f'pages {pages}')

    q_points = pickle.load(open('qpoints_100.dump', 'rb'))

    total = 0

    print("KNN Testing!")

    K = 10
    
    for i in q_points:
        neighbours, pages = root.root.KNN(i, [], K)
        neighbours = np.asarray(neighbours)
        total += pages
        print(f'Point {i}')
        print(f'Neighbours {neighbours}')
        actual = np.asarray([np.asarray((distance(point, i, 2), point[0], point[1])) for point in lst])
        actual = actual[np.argsort(actual[:,0])][:K]
        print(f"Actual {actual}")
        if not (actual == neighbours).all():
            print("OOF")
            raise Exception("OOF")
        for i in actual:
            print(i)
        print(f'Pages {pages}')
    print(f'Average pages is {total/100}\n')

    pdb.set_trace()
    traverse(root.root, ax)
    plt.show()

def test_nd():
    open("rtree_synthetic_nd.txt", "w")
    for dimension in [3,4,5,6]:
        for amount in [1000, 4000, 8000, 16000, 32000, 64000]:
            root = RTree(dimension)
            lst = pickle.load(open(f'synthetic_{amount}_{dimension}d.dump', 'rb'))
            for i in lst:
                root.insert(i)

            # Create qpoints for dimension, between [0, 8000] for all dimensions, and 100 of these points.
            q_points = generate_numbers(0, 8000, 100, dimension)
            # pickle.dump(q_points, open(f'synthetic_qpoints_{dimension}d.dump', 'wb'))
            q_points = pickle.load(open(f'synthetic_qpoints_{dimension}d.dump', 'rb'))
            with open('rtree_synthetic_nd.txt','a') as output:
                output.write(f"{dimension}D:\n")
            
            for K in [1, 5, 10, 50, 100, 500]: 
                count = 0
                total = 0
                for point in q_points:
                    neighbours, pages = root.root.KNN(point, [], K)
                    neighbours = np.asarray(neighbours)
                    total += pages
                    print(f'Point {i}')
                    print(f'Neighbours {neighbours}')
                    actual = np.asarray([np.asarray((distance(i, point, dimension), *i)) for i in lst])
                    actual = actual[np.argsort(actual[:,0])][:K]
                    print(f"Actual {actual}")
                    count += 1
                    if not (actual == neighbours).all():
                        pdb.set_trace()
                        raise Exception(f"{dimension}, {amount}, {K}, {count} = BIG OOF")
                    print(f'Pages {pages}')
                with open('rtree_synthetic_nd.txt', 'a') as output:
                    output.write(f"Average pages for {K} on synthetic points {amount} for dimension {dimension}D: {total/100}.\n")
                print(f'Average pages is {total/100}\n')
def test_3d():
    root = RTree(3)
    lst = generate_numbers(0, 100, 1000, 3)
    # pickle.dump(lst, open("3d_test.dump", 'wb'))
    # lst = pickle.load(open("3d_test.dump", 'rb'))
    # lst = [(0,0,0), (5, 5, 5), (1,2,3), (4,2,3), (0,1,0)]
    lst = [np.asarray(i) for i in lst]
    lst = np.asarray(lst)
    total = 0
    for i in lst:
        root.insert(i)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection = '3d')
    ax.scatter(np.take(lst, 0, 1), np.take(lst, 1, 1), np.take(lst, 2, 1))
    traverse3d(root.root, ax)
    pdb.set_trace()

    q_rects = []
    for i in range(100):
        a = generate_numbers(0, 100, 1, 3)
        b = generate_numbers(0, 100, 1, 3)
        points = np.vstack((a,b))
        mins = [np.min(points[:, i]) for i in range(3)]
        maxs = [np.max(points[:, i]) for i in range(3)]
        q_rect = np.asarray((mins, maxs))
        q_rects.append(q_rect)
    
    count = 0
    for i in q_rects:
        i = np.asarray(i)
        result, pages, ids = root.root.search(i)
        actual = [j for j in lst if overlap(j, i)]
        if len(actual) != len(result):
            print("OOF")
            raise(count)
        print(f"Results: {len(result)}")
        print(f"Actual: {len(actual)}")
        print(f'Pages {pages}')
        count += 1

    q_points = generate_numbers(0, 100, 100, 3)
    pickle.dump(q_points, open('qpoints_3d.dump', 'wb'))

    total = 0

    print("KNN Testing!")

    K = 10
    
    for i in q_points:
        neighbours, pages = root.root.KNN(i, [], K)
        neighbours = np.asarray(neighbours)
        total += pages
        print(f'Point {i}')
        print(f'Neighbours {neighbours}')
        actual = np.asarray([np.asarray((distance(point, i, 3), *point)) for point in lst])
        actual = actual[np.argsort(actual[:,0])][:K]
        print(f"Actual {actual}")
        if not (actual == neighbours).all():
            raise Exception("OOF")
        for i in actual:
            print(i)
        print(f'Pages {pages}')
    print(f'Average pages is {total/100}\n')

    pdb.set_trace()
    plt.show()

    # q_rects = pickle.load(open("query_rectangles_100_100x100.dump", 'rb'))

    # count = 0
    # for i in q_rects:
    #     i = np.asarray(i)
    #     result, pages, ids = root.root.search(i)
    #     print(f"Results: {len(result)}")
    #     total += pages
    #     print(f'Pages {pages}')
    #     count += 1
    # print(f'Average pages {total/100}')

    # result, pages, ids = root.root.search(np.asarray(((20, 20), (45, 45))))
    # print(f'pages {pages}')

    # q_points = pickle.load(open('qpoints_100.dump', 'rb'))

    # total = 0

    # print("KNN Testing!")

    # K = 10
    # 
    # for i in q_points:
    #     neighbours, pages = root.root.KNN(i, [], K)
    #     neighbours = np.asarray(neighbours)
    #     total += pages
    #     print(f'Point {i}')
    #     print(f'Neighbours {neighbours}')
    #     actual = np.asarray([np.asarray((distance(point, i), point[0], point[1])) for point in lst])
    #     actual = actual[np.argsort(actual[:,0])][:K]
    #     print(f"Actual {actual}")
    #     for i in actual:
    #         print(i)
    #     print(f'Pages {pages}')
    # print(f'Average pages is {total/100}\n')

    # pdb.set_trace()
    # traverse(root.root, ax)
    # plt.show()
if __name__ == '__main__':
    # pdb.set_trace()
    # root = RTree()
    # lst = generate_numbers(0, 100, 100)
    # for i in range(5):
    #     pdb.set_trace()
    #     root.insert(lst[i])
    # test_synthetics()
    test_images()
    # test_1000()
    # test_3d()
    # test_nd()
    # test_long_beach()
    # pdb.set_trace()
    # root = RTree()
    # root.insert((2,3))
    # root.insert((1,5))
    # root.insert((10, 15))
    # root.insert((1000, 140))
    # root.insert((30, 20))
    # lst = generate_numbers(0, 100, 20)
    # pickle.dump(lst, open('rtree_20.dump', 'wb'))
    # lst = pickle.load(open('rtree_20.dump', 'rb'))
    
    # for i in q_points:
    #     neighbours, pages = root.root.KNN(i, [], 3)
    #     neighbours = np.asarray(neighbours)
    #     total += pages
    #     print(f'Point {i}')
    #     print(f'Neighbours {neighbours}')
    #     print(f'Pages {pages}')
    # print(f'Average pages is {total/100}')
    # result, pages = root.root.search(np.asarray(((48, 3), (67, 18))))
    # actual = [point for point in lst if overlap(point, ((48, 3), (67, 18)))]
    # pdb.set_trace()

    # for i in root.root.children:
    #     print(min_max_dist(np.asarray((10, 20)), i.rectangle))

    # print(pages)

    # for i in result:
    #     found = False
    #     for j in range(len(actual)):
    #         if (actual[j] == i).all():
    #             found = True
    #     if not found:
    #         raise Exception("xd")
    #         print(i)
    #         print("was not found!")


