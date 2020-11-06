from typing import Tuple, List, Optional
import numpy as np
from data_generation import cluster, generate_numbers
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import pdb
import pickle

M =5 # Arbitrarily set M.

def distance(point1, point2):
    """
    Calculate the distance between 2 points
    point1: 1x2 array representing first point
    point2: 1x2 array representing second point
    """
    p1_x, p1_y = point1
    p2_x, p2_y = point2

    return (p1_x - p2_x)**2 + (p1_y - p2_y)**2


def bounding_rectangle(points):
    """
    Points is a np array of N points of 2 elements. In other words, this is a (N, 2) np array.
    Returns the bounding rectangle that closely encloses all points in points. This is in format [(lower_left_x, lower_left_y), (top_right_x, top_right_y)]
    """
    X = np.take(points, 0, 1)
    Y = np.take(points, 1, 1)
    max_x = np.amax(X)
    max_y = np.amax(Y)
    min_x = np.amin(X)
    min_y = np.amin(Y)
    return np.array([(min_x, min_y), (max_x, max_y)])

def min_max_dist(point, rectangle):
    """
    point is a 1x2 np array. Represents a point
    rectangle is a 2x2 np array. Represents the bottom-left and top-right corners respectively

    Returns the "MINMAXDIST" as required by the paper.
    """
    rect_points = np.stack((np.take(rectangle, 0, 1), np.take(rectangle, 1, 1)))

    S = 0
    for i in range(2):
        interval = rect_points[i]
        rM = interval[0] if point[i] >= (interval[0] + interval[1])/2 else interval[1]
        S += (point[i] - rM) ** 2

    minimum = float('inf')
    for i in range(2):
        interval = rect_points[i]
        rM = interval[0] if point[i] >= (interval[0] + interval[1])/2 else interval[1]
        rm = interval[1] if point[i] >= (interval[0] + interval[1])/2 else interval[0]
        minimum = min(minimum, S - (point[i] - rM) ** 2 + (point[i] - rm) ** 2)
    return minimum

def min_dist(point, rectangle):
    """
    point is a 1x2 np array. Represents a point
    rectangle is a 2x2 np array. Represents the bottom-left and top-right corners respectively

    Returns the "minimum_distance" as required by the paper.
    """
    rect_points = np.stack((np.take(rectangle, 0, 1), np.take(rectangle, 1, 1)))

    count = 0
    
    for i in range(2):
        interval = rect_points[i]
        r = 0
        if point[i] < interval[0]:
            r = interval[0]
        elif point[i] > interval[1]:
            r = interval[1]
        else:
            r = point[i]
        count += (point[i] - r)**2
    return count

def adjust_rectangle(node):
    """
    Given an RTree_node, return the bounding rectangle that contains all its children rectangles.
    """
    rectangles = np.empty([1,2])
    for i in node.children:
        rectangles = np.concatenate((rectangles, i.rectangle))
    
    rectangles = np.delete(rectangles, 0 ,0)
    return bounding_rectangle(rectangles)

def overlap(point, rect):
    """
    Find if point overlaps rectangle rect.

    point is a 1x2 array representing a point.
    rect is a 2x2 array representing first the bottom left corner, then the top right corner.
    """
    bottom_left, top_right = rect
    p_x, p_y = point

    min_x, min_y = bottom_left
    max_x, max_y = top_right
    
    return min_x <= p_x <= max_x and min_y <= p_y <= max_y

def overlap_rectangles(rect1, rect2):
    """
    Determine if two rectangles overlap.

    rect1 and rect2 are 2 x 2 arrays.
    """
    x_1 = np.take(rect1, 0, 1)
    y_1 = np.take(rect1, 1, 1)

    x_2 = np.take(rect2, 0, 1)
    y_2 = np.take(rect2, 1, 1)

    points_1 = [(i,j) for i in x_1 for j in y_1]
    points_2 = [(i,j) for i in x_2 for j in y_2]

    for i in points_1:
        if overlap(i, rect2):
            return True
    for i in points_2:
        if overlap(i, rect1):
            return True
    return False
    

class RTree_node:
    """
    This class represents the internal nodes of an R-tree
    The children are going to be addresses of a lower node. The rectangle will cover the entire points encapsulated by the children.
    """
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
                min_x = rect[0][0]
                min_y = rect[0][1]
                max_x = rect[1][0]
                max_y = rect[1][1]
                centroid = ((max_x + min_x)/2, (max_y + min_y)/2)
                lst.append(centroid)
                mapping[centroid] = i

            arr = np.array(lst)
            lst1, lst2, kmeans = cluster(arr, M)
            node2 = RTree_node([], [self.children[mapping[tuple(i)]] for i in lst2], self.parent, [])

            self.children = [self.children[mapping[tuple(i)]] for i in lst1]
            self.data_points = []
            node1 = self
            return node1, node2
    
    def search(self, rectangle):
        """
        Search for all points within rectangle
        
        Returns a list of all points found within rectangle.
        """
        # Start at root
        ans = []
        pages = 1 # Have to read self once.

        if self.children == []:
            # We are a leaf node.
            for i in self.data_points:
                if overlap(i, rectangle):
                    ans.append(i)
        else:
            # Loop through children
            # Let's use DFS for this.
            for child in self.children:
                rect = child.rectangle
                if overlap_rectangles(rect, rectangle):
                    a, p = child.search(rectangle)
                    ans.extend(a)
                    pages += p
        return ans, pages

    def KNN(self, point, nearest, k):
        """
        point: 1x2 array representing the point
        nearest: List representing the nearest k positions so far.
        k: The number of nearest neighbours to find.
        """
        pages = 0
        if self.children == []:
            # We are a leaf
            for i in self.data_points:
                dist = distance(i, point)
                dist_entry = np.insert(i, 0, dist)
                if len(nearest) < k:
                    nearest.append(dist_entry)
                else:
                    # Sort the objects found in the children.
                    nearest = np.asarray(nearest)
                    nearest = np.concatenate((nearest,dist_entry.reshape(1,3)), 0)
                    nearest = nearest[np.argsort(nearest[:,0])][:k]
                    nearest = list(nearest)
            pages += 1
        else:
            # We aren't a leaf.
            branches = []
            for i in self.children:
                branches.append((min_max_dist(point, i.rectangle), min_dist(point, i.rectangle), i))
                # sort by min_max_dist
            branches = sorted(branches)
            branches = self._prune(branches, nearest, k)

            i = 0
            while i < len(branches):
                child = branches[i][2]
                nearest, p = child.KNN(point, nearest, k)
                pages += p
                branches = self._prune(branches, nearest, k)

                i += 1
        return nearest, pages

    def _prune(self, branches, nearest, k):
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
                for j, element in enumerate(lst):
                    if j == i:
                        continue
                    if element[1] > curr_min_max_dist or element[1] > max_nearest:
                        prune_set.add(j)
            if len(prune_set) == 0:
                # Finished pruning.
                break
            else:
                # Something to prune, need to prune and rerun
                branches = [e for i, e in enumerate(branches) if i not in prune_set]
        return branches
        
class RTree:
    """
    Implementation of an R-Tree. It only stores the root.
    """
    def __init__(self):
        self.root = RTree_node([], [], None)


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
                p2.rectangle = adjust_rectangle(p2)
            p1.rectangle = adjust_rectangle(p1)
           
            return self.adjust_tree(p1, p2)

            
        pass
    def chooseLeaf(self, start: RTree_node, point: Tuple[float, float]):
        curr = start
        if curr.children == []:
            return curr
        else:
            min_increase = float("inf")
            min_node = None
            for i in curr.children:
                rect = i.rectangle
                min_x = rect[0][0]
                min_y = rect[0][1]
                max_x = rect[1][0]
                max_y = rect[1][1]

                curr_area = (max_x - min_x) * (max_y - min_y)
                new_area = (max(max_x, point[0]) - min(min_x, point[0])) * (max(max_y, point[1]) - min(min_y, point[1]))
                if new_area - curr_area < min_increase:
                    min_node = i
                    min_increase = new_area - curr_area
                # TODO: Check if we're choosing the right subtree.
            return self.chooseLeaf(min_node, point)

    def insert(self, point: Tuple[float, float]):
        leaf = self.chooseLeaf(self.root, point)
        n1, n2 = leaf.insert_to_leaf(np.array(point))
        n1, n2 = self.adjust_tree(n1, n2)
        if n2 and n1.parent == None:
            # We had to split at root.
            self.root = RTree_node([], [n1, n2], None, []) 
            n1.parent = self.root
            n2.parent = self.root

                    

def traverse(root, fig):
    if root.children == []:
        print(len(root.data_points))
    for i in root.children:
        
        rect = i.rectangle
        min_x = rect[0][0]
        min_y = rect[0][1]
        max_x = rect[1][0]
        max_y = rect[1][1]
        ax.add_patch(Rectangle(xy=(min_x, min_y), width=max_x-min_x, height=max_y-min_y, fill=False, color='blue'))
        traverse(i, ax)

if __name__ == '__main__':
    root = RTree()
    # root.insert((2,3))
    # root.insert((1,5))
    # root.insert((10, 15))
    # root.insert((1000, 140))
    # root.insert((30, 20))
    # lst = generate_numbers(0, 100, 20)
    # pickle.dump(lst, open('rtree_20.dump', 'wb'))
    # lst = pickle.load(open('rtree_20.dump', 'rb'))
    lst = pickle.load(open('data_100.dump', 'rb'))

    fig = plt.figure()
    ax = fig.add_subplot(111)
    for i in lst:
        root.insert(i)
    x = np.take(lst,0, 1)
    y = np.take(lst, 1, 1)
    ax.scatter(x,y, color = 'red')

    q_points = generate_numbers(0, 100, 10)
    # pickle.dump(q_points, open('qpoints_10.dump', 'wb'))
    q_points = pickle.load(open('qpoints_10.dump', 'rb'))

    traverse(root.root, ax)
    
    for i in q_points:
        neighbours, pages = root.root.KNN(i, [], 3)
        neighbours = np.asarray(neighbours)
        print(f'Point {i}')
        print(f'Neighbours {neighbours}')
        print(f'Pages {pages}')

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
    plt.show()


