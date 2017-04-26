from .Bagging import Bagging


class RandomRoughForest(object):
    """
    For each of the trees in the forest, using rough set
    to get the reduct of universe(data) subspace, train the
    forest with different classifier,the classifier result
    can be conduct from majority voting.
    """
    def __init__(self, train_data, strategy):
        # self.bagging = Bagging(train_data, strategy)
        self.forest = []
        self.bagging = Bagging(train_data, strategy)
        self.forest.extend(self.bagging.get_bags())

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

    def get_reduced(self):
        result = {}
        for tree in self.forest:
            dependency_dict = tree.get_reduced()
            for k in dependency_dict.keys():
                if k not in result:
                    result[k] = dependency_dict[k]
                else:
                    result[k] += dependency_dict[k]
        return result

