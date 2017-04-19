import random
import math
import numpy as np

from core.BaseRoughSet import BaseRoughSet


class Bagging(object):

    def __init__(self, train_data):
        self.data = train_data
        self.row, self.col = train_data.shape
        self.strategy = None

    def get_bag(self):
        """
        Select and return data with target cols & rows
        """
        row_idx, col_idx = self._get_rand_subspace()
        bag = self.data[row_idx, :][:, col_idx]

        # Can be consuming
        return Bag(bag)

    def _get_rand_subspace(self):

        loop_times = self.row
        rows = set()
        for i in range(loop_times):
            rows.add(random.randint(0, loop_times - 1))
        row_idx = np.array(list(rows))

        cols = random.sample(range(0, self.col - 1), math.ceil(math.sqrt(self.col)))
        col_idx = np.array(cols)

        return row_idx, col_idx


class Bag(object):

    def __init__(self, bag):
        self.bag = bag

    def get_reduced(self):
        bag = self.bag
        rough_set = BaseRoughSet(bag[:, :-1], bag[:, -1])
        print(rough_set.get_reduced())
        # return rough_set.get_reduced()

    def train(self):
        pass

    def classify(self):
        pass


