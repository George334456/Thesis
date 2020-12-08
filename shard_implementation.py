import numpy as np
import pdb
from rtree import bounding_rectangle

class Shard_index:
    def __init__(self, identification):
        """
        Class that stores data within a page
        """
        self.shards = {}
        self.index = identification

    def append(self, position, point):
        """
        Append point to shard at position position in this shard_index

        Position is an integer
        point is a 1x2 np array
        """
        if position not in self.shards:
            self.shards[position] = Shard(position)
        self.shards[position].append(point)

    def get_keys(self):
        """
        Returns a list of all keys in this shard_index
        """
        return list(self.shards.keys())

    def get(self, position):
        """
        Get the shard at position position.
        """
        if position in self.shards:
            return self.shards[position]
        else:
            return None

    def get_id(self):
        return self.index

    def calculate(self):
        for i in self.shards:
            self.shards[i].calculate()

class Rectangle:
    def __init__(self, identification, bounding_rect):
        self.identification = identification
        self.rectangle = np.asarray(bounding_rect)
        self.lower_mapping = None
        self.upper_mapping = None

    def set_mappings(self, lower, upper):
        self.lower_mapping = lower
        self.upper_mapping = upper

class Shard:
    def __init__(self, identification):
        """
        Initialize an empty shard at position identification.
        """
        self.lower_mapping = None
        self.upper_mapping = None
        self.bounding_rectangles = []
        self.data = []
        self.id = identification

    def append(self, point):
        """
        Appends point to this shard.

        point is a 1x2 array
        """
        self.data.append(point)

    def get_data(self):
        return self.data

    def get_bounding_rectangles(self):
        return self.bounding_rectangles

    def calculate(self):
        """
        Calculate the bounding rectangle, and remember the lower/upper mapping
        """
        points = np.asarray(self.data)
        self.lower_mapping = np.min(points[:,0])
        self.upper_mapping = np.max(points[:,0])
        count = 0
        for i in range(int(self.lower_mapping), int(self.upper_mapping) + 1):
            point_lst = [point[1:] for point in points if i <= point[0] < i + 1]
            if point_lst:
                rect = Rectangle(count, bounding_rectangle(point_lst))
                rect.set_mappings(i, i+1)
                self.bounding_rectangles.append(rect)
                count += 1

    def get_page(self, page, psi):
        """
        From this shard's data, get the page specified.

        page is an integer representing the page that we want.
        psi is an integer representing how many data points to put into a page.
        """
        start = page * psi
        result = []
        for i in range(start, min(start + psi, len(self.data))):
            result.append(self.data[i])
        return result

class Cell:
    def __init__(self, min_map, bounding_rectangle):
        """
        Creates a cell with the mapping range it contains. More specifically, this describes the cell with all mappingings [min_map, min_map + 1)

        Creates a bounding_rectangle around it. This bounding rectangle is a [dimension x 2] array
        """
        self.mapping = min_map
        self.bounding_rectangle = np.asarray(bounding_rectangle)
        self.shards = set()
    
    def add_shard(self, shard):
        """
        Add a shard to the Cell. We don't use a particular shard object, but an (index, shard_id) pair to represent the shard.

        shard is a 2 length tuple. The first element is an index, and the second element is the shard_id.
        """
        self.shards.add(shard)
