from core.BaseRoughSet import BaseRoughSet
from core.strategy.RandomSubspace import RandomSubspace
from core.strategy.ReducedSubspace import ReducedSubspace


class Bagging(object):

    def __init__(self, train_data, subspace_strategy):
        self.data = train_data
        self.row = train_data.shape[0]
        self.col = train_data.shape[1] - 1
        self.bags = []

        # todo: reflection implement version
        if subspace_strategy == "RandomSubspace":
            self.subspace_strategy = RandomSubspace.get_subspace
        elif subspace_strategy == "ReducedSubspace":
            self.subspace_strategy = ReducedSubspace.get_subspace
        else:
            raise NotImplementedError(subspace_strategy + "has not implemented yet.")

        self.reduced_dict = BaseRoughSet(train_data[:, :-1], train_data[:, -1]).get_reduced_from_core()

        for k, v in self.reduced_dict:
            row_idx, col_idx = self.subspace_strategy((self.row, self.col))
            if col_idx is None:
                col_idx = list(k).copy()
            col_idx.append(self.col)
            self.bags.append(Bag(train_data, row_idx, col_idx))

    def get_bags(self):
        return self.bags

    def train(self):
        pass

    def classify(self):
        pass


class Bag(object):

    def __init__(self, data, row_idx, col_idx):
        self.row_idx = row_idx
        self.col_idx = col_idx
        self.data = data[row_idx, :][:, col_idx]






