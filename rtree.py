from typing import Tuple, List, Optional
import numpy as np
from data_generation import cluster
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import pdb

M =2 # Arbitrarily set M.

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

def adjust_rectangle(node):
    """
    Given an RTree_node, return the bounding rectangle that contains all its children rectangles.
    """
    rectangles = np.empty([1,2])
    for i in node.children:
        rectangles = np.concatenate((rectangles, i.rectangle))
    
    rectangles = np.delete(rectangles, 0 ,0)
    return bounding_rectangle(rectangles)
    

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
        # pdb.set_trace()
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
        print("splitting time")
        if self.children == []:
            # This is a leaf node
            lst = []
            for i in self.data_points:
                lst.append(i)
            arr = np.array(lst)
            lst1, lst2, kmeans = cluster(arr, M)
            print("KEKW")
            print(lst1)
            print(bounding_rectangle(lst1))
            print(lst2)
            print(bounding_rectangle(lst2))
            # Node 1 modifies itself (to avoid looking it up)
            self.rectangle = bounding_rectangle(lst1)
            self.children = []
            self.data_points = list(lst1)
            print("Printing lst1 {}".format(self.data_points))
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
                pdb.set_trace()
                centroid = ((max_x + min_x)/2, (max_y + min_y)/2)
                lst.append(centroid)
                mapping[centroid] = i

            arr = np.array(lst)
            lst1, lst2, kmeans = cluster(arr, M)
            pdb.set_trace()
            node2 = RTree_node([], [self.children[mapping[tuple(i)]] for i in lst2], self.parent, [])

            self.children = [self.children[mapping[tuple(i)]] for i in lst1]
            self.data_points = []
            node1 = self
            return node1, node2

        
                # This is now an internal node IE there are rectangles that matter.


            
        pass

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
            pdb.set_trace()
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
            print("Hi")
            # We had to split at root.
            self.root = RTree_node([], [n1, n2], None, []) 
            n1.parent = self.root
            n2.parent = self.root

def traverse(root, fig):
    for i in root.children:
        
        rect = i.rectangle
        min_x = rect[0][0]
        min_y = rect[0][1]
        max_x = rect[1][0]
        max_y = rect[1][1]
        ax.add_patch(Rectangle(xy=(min_x, min_y), width=max_x-min_x, height=max_y-min_y, fill=False, color='blue'))
        print("Rect")
        print(i.rectangle)
        
        print(i.data_points)
        traverse(i, ax)
        

if __name__ == '__main__':
    root = RTree()
    root.insert((2,3))
    root.insert((1,5))
    root.insert((10, 15))
    root.insert((1000, 140))
    pdb.set_trace()
    root.insert((30, 20))
    fig = plt.figure()
    ax = fig.add_subplot(111)
    a = ([2, 1, 10, 1000, 30],[3, 5, 15,140, 20])
    ax.scatter(a[0],a[1], color = 'red')
    pdb.set_trace()
    traverse(root.root, ax)
    plt.show()


