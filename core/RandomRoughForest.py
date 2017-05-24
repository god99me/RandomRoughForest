from .Bagging import Bagging
from sklearn.metrics import accuracy_score


class RandomRoughForest(object):

    def __init__(self, data, radius_range, n_trees):
        # self.bagging = Bagging(train_data, strategy)
        self.forest = []
        self.data = data
        self.oob = list(range(data.shape[0]))  # pass ref

        for i in range(n_trees):
            self.forest.append(Bagging(data, radius_range, self.oob))

    def train(self):
        for tree in self.forest:
            tree.train()

    def classify(self):
        test_set = self.data.iloc[self.oob]
        predicted_list = []
        for row in test_set.values:
            predictions = [tree.classify(row[ :-1]) for tree in self.forest]
            predicted = max(set(predictions), key=predictions.count)
            predicted_list.append(predicted)
        return predicted_list

    def evaluate(self):
        predicted_list = self.classify()
        actual = self.data.iloc[self.oob, :-1]
        accuracy = RandomRoughForest.accuracy_metric(actual, predicted_list)
        # return accuracy_score(Y_validation, predictions)
        return accuracy

    @staticmethod
    def accuracy_metric(actual, predicted):
        correct = 0
        for i in range(len(actual)):
            if actual[i] == predicted[i]:
                correct += 1
        return correct / float(len(actual))
