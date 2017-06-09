import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt

from pandas.tools.plotting import scatter_matrix
from sklearn import cross_validation
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

from core.RandomRoughForest import RandomRoughForest

import core.BaseRoughSet as RS
from core.Preprocessing import normalize

from sklearn.datasets import load_iris


def main():

    data_source = get_data_source()

    for name, data, radius_range in data_source:
        # test_calc_core(data, name, radius_range)
        # test_calc_red(data, name, radius_range)
        test_calc_accuracy(data, name, radius_range)


def test_calc_accuracy(data, name, radius_range):
    data = pd.DataFrame(data)
    print(name, "shape is", data.shape, "\n")
    print(name, "first 5 rows are: ")
    print(data.iloc[:, :-1].head(5).values, "\n")
    print("describe %s :" % name)
    print(data.iloc[:, :-1].describe(), "\n")
    print("group by classification target: ")
    print(data.groupby(data.shape[1] - 1).size(), "\n")
    # split-out validation dataset
    array = data.values
    X_train = array[:, :data.shape[1] - 1]
    Y_train = array[:, data.shape[1] - 1]
    # validation_size = 0.20
    # seed = 7
    # X_train, X_validation, Y_train, Y_validation = \
    #     cross_validation.train_test_split(X, Y, test_size=validation_size, random_state=seed)
    # Test options and evaluation metric
    num_folds = 10
    num_instances = len(X_train)
    seed = 7
    scoring = 'accuracy'
    # Spot Check Algorithms
    models = []
    models.append(('LR', LogisticRegression()))
    models.append(('LDA', LinearDiscriminantAnalysis()))
    models.append(('KNN', KNeighborsClassifier()))
    models.append(('CART', DecisionTreeClassifier()))
    models.append(('NB', GaussianNB()))
    models.append(('SVM', SVC()))
    models.append(('RF', RandomForestClassifier(n_estimators=50)))
    # evaluate each model in turn
    results = []
    names = []
    for name, model in models:
        kfold = cross_validation.KFold(n=num_instances, n_folds=num_folds, random_state=seed)
        cv_results = cross_validation.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
        results.append(cv_results)
        names.append(name)
        msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
        print(msg)

    scores = RandomRoughForest.k_fold_evaluate(data, radius_range, 50)
    print("%s: %f (%f) \n" % ("NRRF", np.mean(scores), np.std(scores)))


def test_calc_core(data, name, radius_range):
    core_list = []
    print("Processing: %s \n" % name)
    for radius in radius_range:
        result = RS.partition_all(data, radius)
        print("partition on raidus %f completed." % radius)
        core = RS.calc_core(result)
        print("core is ", core)
        core_list.append(len(core))
    print(name, core_list, "\n")


def test_calc_red(data, name, radius_range):
    print("Processing: %s \n" % name)
    for radius in radius_range:
        result = RS.partition_all(data, radius)

        core = RS.calc_core(result)
        # core_list.append(len(core))
        print("core: ", core)

        core_pos_set = RS.calc_pos_set(result.iloc[:, list(core)])

        red_set = set()
        for i in range(5):
            red = RS.calc_red(result, core, core_pos_set, 0.15)
            if len(red) > 0:
                red_set.add(tuple(red))

        if len(red_set) == 0:
            print("empty reduct")
        else:
            min_len = data.shape[1]
            min_len_red = None
            for red in red_set:
                temp_len = len(red)
                if temp_len < min_len:
                    min_len = temp_len
                    min_len_red = red

            print("reduct: ", min_len_red)

            # rrf = RandomRoughForest(data, radius_range, 50)
            # rrf.train()
            # score = rrf.evaluate()
            # print(score)


def get_data_source():
    iris = load_iris()
    iris_data = normalize(iris.data)
    iris_target = pd.Series(iris.target)
    iris = pd.concat([iris_data, iris_target], axis=1)
    iris.columns = range(iris.shape[1])
    yield ("iris", iris, RS.float_range(0.6, 2, 10))
    # # 0.5 ~ 2  7.6 ~ 8.8 8.8 ~ 10

    wine = pd.read_csv("datasets/wine/wine.data")
    target = wine.pop("14")
    wine.insert(13, "14", target)
    wine.columns = range(wine.shape[1])
    wine_data = normalize(wine.iloc[:, :-1])
    wine.iloc[:, :-1] = wine_data
    yield ("wine", wine, RS.float_range(0.72, 0.78, 10))
    # 0.7 ~ 0.8

    sonar = pd.read_csv("datasets/sonar.all-data", header=None)
    sonar_data = normalize(sonar.iloc[:, :-1])
    sonar.iloc[:, :-1] = sonar_data

    category = pd.Categorical(sonar.iloc[:, -1])
    sonar.iloc[:, -1] = category.codes

    sonar.columns = range(sonar.shape[1])
    yield ("sonar", sonar, RS.float_range(0.3, 0.6, 10))
    # 0.3 ~ 0.6

    # bank = pd.read_csv("datasets/bank/bank-full.csv", sep=";")
    # bank.columns = range(bank.shape[1])
    #
    # count = 0
    # for j in range(bank.shape[1]):
    #     dtype_name = bank.iloc[:, j].dtype.name
    #     if dtype_name.find("object") == -1:
    #         continue
    #     category = pd.Categorical(bank.iloc[:, j])
    #     bank.iloc[:, j] = category.codes
    #     count += 1
    # yield ("bank", bank)


if __name__ == '__main__':
    main()

