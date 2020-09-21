from typing import Tuple, List, Optional

M = 4 # Arbitrarily set M.

class RTree_node:
    """
    This class represents the internal nodes of an R-tree
    The children are going to be addresses of a lower node. The rectangle will cover the entire points encapsulated by the children.
    """
    def __init__(self, rectangle: List, children, parent):
        """Sets the stuff for a node

        If a node has children, then it is not a leaf, and the data_point position has no meaning.
        If a node doesn't have children, then it is considered a leaf, and the data_point data can be accessed
        Parent is the parent Node. For the root, Parent is None
        """
        self.parent = parent
        self.rectangle = rectangle
        self.children = children
        self.data_points = []

    def insert_to_leaf(self, point: Tuple[float, float]):
        """
        Inserts a point into the node
        """
        self.data_points.append(point)
        if len(self.data_points) > M:
            self.split_node() 

    def split_node(self):
        print("splitting time")
        pass

class RTree:
    """
    Implementation of an R-Tree. It only stores the root.
    """
    def __init__(self):
        self.root = RTree_node([], [], None)

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
                min_y = rect[1][0]
                max_x = rect[0][1]
                max_y = rect[1][1]

                curr_area = (max_x - min_x) * (max_y - min_y)
                new_area = (max(max_x, point[0]) - min(min_x, point[0]) * (max(max_y, point[1]) - min(min_y, point[1])))
                if new_area - curr_area < min_increase:
                    min_node = i
                    min_increase = new_area - curr_area
                # TODO: Check if we're choosing the right subtree.
            return self.chooseLeaf(min_node, point)

    def insert(self, point: Tuple[float, float]):
        leaf = self.chooseLeaf(self.root, point)
        leaf.insert_to_leaf(point)

if __name__ == '__main__':
    root = RTree()
    root.insert((2,3))
    root.insert((1,5))
    root.insert((10, 15))
    root.insert((1000, 140))
    root.insert((30, 20))
    print(root.root.children)
    print(root.root.data_points)
