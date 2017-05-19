from .Bagging import Bagging


class RandomRoughForest(object):

    def __init__(self, data, types, radius_range, n_trees):
        # self.bagging = Bagging(train_data, strategy)
        self.forest = []
        self.data = data
        self.oob = list(range(data.shape[0]))  # pass ref

        for i in range(n_trees):
            self.forest.append(Bagging(data, types, radius_range, self.oob))

    def train(self):
        for tree in self.forest:
            tree.train()

    def classify(self):
        votes = {}

        for tree in self.forest:
            label = tree.classify(self.data.iloc[self.oob])
            if label in votes:
                votes[label] += 1
            else:
                votes[label] = 1

        majority = None
        max_cnt = 0
        for key in votes:
            if votes[key] > max_cnt:
                majority = key
                max_cnt = votes[key]

        return majority
