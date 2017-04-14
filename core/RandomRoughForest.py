from .Bagging import Bagging


class RandomRoughForest(object):
    """
    For each of the trees in the forest, using rough set
    to get the reduct of universe(data) subspace, train the
    forest with different classifier,the classifier result
    can be conduct from majority voting.
    """
    def __init__(self, train_data, num_of_trees=100):
        # self.data = train_data
        self.num_of_trees = num_of_trees
        self.bagging = Bagging(train_data)
        self.forest = []

        for i in range(num_of_trees):
            self.forest.append(self.bagging.get_bag())

    def train(self):
        for tree in self.forest:
            tree.train()

    def classify(self, test_set):
        votes = {}

        for tree in self.forest:
            label = tree.classify(test_set)
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

