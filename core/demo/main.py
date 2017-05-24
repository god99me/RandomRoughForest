import pandas as pd
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

from sklearn.datasets import load_iris


def main():

    data_source = get_data_source()

    for name, data in data_source:
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
        models.append(('RF',  RandomForestClassifier(n_estimators=100)))

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
        print("\n")


def get_data_source():
    data_list = []
    iris = load_iris()
    iris_data = pd.DataFrame(iris.data)
    iris_target = pd.Series(iris.target)
    iris = pd.concat([iris_data, iris_target], axis=1)
    iris.columns = range(iris.shape[1])
    data_list.append(("iris", iris))

    wine = pd.read_csv("../datasets/wine/wine.data")
    target = wine.pop("14")
    wine.insert(13, "14", target)
    wine.columns = range(wine.shape[1])
    data_list.append(("wine", wine))

    sonar = pd.read_csv("../datasets/sonar.all-data")
    sonar.columns = range(sonar.shape[1])
    data_list.append(("sonar", sonar))

    bank = pd.read_csv("../datasets/bank/bank-full.csv", sep=";")
    bank.columns = range(bank.shape[1])

    count = 0
    for j in range(bank.shape[1]):
        dtype_name = bank.iloc[:, j].dtype.name
        if dtype_name.find("object") == -1:
            continue
        category = pd.Categorical(bank.iloc[:, j])
        bank.iloc[:, j] = category.codes
        count += 1

    data_list.append(("bank", bank))
    return data_list


if __name__ == '__main__':
    main()

