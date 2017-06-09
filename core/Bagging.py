import random
from math import sqrt

from sklearn import tree

import core.BaseRoughSet as RS


class Bagging(object):

    def __init__(self, data, radius_range, oob_set):
        self.samples = Bagging.fetch_samples(data.shape[0])

        self.oob_set = oob_set
        self.update_oob_set(oob_set, self.samples)
        self.samples = list(self.samples)
        self.features = Bagging.fetch_features(data, radius_range)
        self.data = data
        # self.data = Bagging.fetch_data(data, self.samples, self.features)

        self.clf = tree.DecisionTreeClassifier()

    def classify(self, test_set):
        prediction = self.clf.predict(test_set.iloc[self.features].reshape(1, -1))
        return prediction[0]

    def train(self):
        X_train = Bagging.fetch_data(self.data, self.samples, self.features)
        Y_train = Bagging.fetch_decision(self.data, self.samples)
        self.clf.fit(X_train, Y_train)

    @staticmethod
    def fetch_features(data, radius_range):
        radius = random.choice(radius_range)
        result = RS.partition_all(data, radius)
        core = RS.calc_core(result)
        core_pos_set = RS.calc_pos_set(result.iloc[:, list(core)])

        # red_set = set()
        # for i in range(5):
        red = RS.calc_red(result, core, core_pos_set, 0.1)
        if len(red) == 0:
            print("no feature selected in radius equals to ", radius)
            n_features = data.shape[1]
            red = random.sample(range(n_features), round(sqrt(n_features)))
        # red_set.add(tuple(red))

        return list(red)

    @staticmethod
    def fetch_samples(n_rows):
        """Get random rows with replacement"""
        selected_rows = set()
        for i in range(n_rows):
            selected_rows.add(random.randint(0, n_rows - 1))
        return selected_rows

    @staticmethod
    def fetch_data(data, samples, features):
        return data.iloc[samples, features]

    @staticmethod
    def fetch_decision(data, samples):
        n_cols = data.shape[1]
        return data.iloc[samples, n_cols - 1]

    def update_oob_set(self, oob_set, samples):
        oob_set.difference_update(samples)
