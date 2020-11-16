import numpy as np
import pdb
from rtree import bounding_rectangle

class Shard_index:
    def __init__(self):
        """
        Class that stores data within a page
        """
        self.shards = {}

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

    def calculate(self):
        for i in self.shards:
            self.shards[i].calculate()

class Shard:
    def __init__(self, identification):
        """
        Initialize an empty shard at position identification.
        """
        self.lower_mapping = None
        self.upper_mapping = None
        self.bounding_rectangle = None
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

    def calculate(self):
        """
        Calculate the bounding rectangle, and remember the lower/upper mapping
        """
        points = np.asarray(self.data)
        self.lower_mapping = np.min(points[:,0])
        self.upper_mapping = np.max(points[:,0])
        self.bounding_rectangle = bounding_rectangle(points)

