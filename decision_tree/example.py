from sklearn.datasets import load_iris
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from decision_tree_pybind import DecisionTree
import numpy as np


"""
class DecisionTreeClassifierNative(object):
    def __init__(self, random_state=1, max_count=1000):
        self.seed = random_state;
        self.max_count = max_count
        self.predictor = None

    def fit(self, x, y):
        self.predictor = decision_tree_pybind.DecisionTree(x, y, self.seed, self.max_count)

    def predict(self, x):
        return self.predictor.predict(x)
"""

iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.3, random_state=4)

dtc = DecisionTreeClassifier(random_state=0)
dtc.fit(X_train, y_train)
y_predicted = dtc.predict(X_test)
print(classification_report(y_test, y_predicted))

dt = DecisionTree(random_state=0)
dt.fit(X_train, y_train)
y_predicted = dt.predict(X_test)
print(type(y_predicted))
print(len(y_predicted), len(y_test))
print(classification_report(y_test, y_predicted))
