from .Bagging import Bagging
from sklearn.cross_validation import KFold


class RandomRoughForest(object):

    def __init__(self, data, radius_range, n_trees):
        # self.bagging = Bagging(train_data, strategy)
        self.forest = []
        self.data = data
        self.oob = set(range(data.shape[0]))  # pass ref

        for i in range(n_trees):
            self.forest.append(Bagging(data, radius_range, self.oob))

    def train(self):
        for tree in self.forest:
            tree.train()

    def classify(self):
        test_set = self.data.iloc[list(self.oob), :-1]
        predicted_list = []
        for i in range(test_set.shape[0]):
            predictions = [tree.classify(test_set.iloc[i]) for tree in self.forest]
            predicted = max(set(predictions), key=predictions.count)
            predicted_list.append(predicted)
        return predicted_list

    @staticmethod
    def k_fold_evaluate(data_set, radius_range, n_trees):
        n_samples, n_features = data_set.shape
        kf = KFold(n_samples, n_folds=3, shuffle=True)
        X = data_set.iloc[:, :-1]
        Y = data_set.iloc[:, -1]
        accuracies = []
        for train_index, test_index in kf:
            x_test, y_test = X.iloc[test_index], Y.iloc[test_index]

            nrrf = RandomRoughForest(data_set.iloc[train_index], radius_range, n_trees)
            nrrf.train()

            predicted_list = []
            for i in range(len(test_index)):
                predictions = [tree.classify(x_test.iloc[i]) for tree in nrrf.forest]
                predicted = max(set(predictions), key=predictions.count)
                predicted_list.append(predicted)

            accuracy = RandomRoughForest.accuracy_metric(list(y_test), predicted_list)
            accuracies.append(accuracy)
        return accuracies

    def evaluate(self):
        predicted_list = self.classify()
        actual = self.data.iloc[list(self.oob), -1]
        accuracy = RandomRoughForest.accuracy_metric(list(actual), predicted_list)
        # return accuracy_score(Y_validation, predictions)
        return accuracy

    @staticmethod
    def accuracy_metric(actual, predicted):
        correct = 0
        for i in range(len(actual)):
            if actual[i] == predicted[i]:
                correct += 1
        return correct / float(len(actual))
