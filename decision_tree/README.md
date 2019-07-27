# Usage example

```
from sklearn.datasets import load_iris
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from decision_tree_pybind import DecisionTree
import numpy as np

iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.3, random_state=4)

dtc = DecisionTreeClassifier(random_state=0)
dtc.fit(X_train, y_train)
y_predicted = dtc.predict(X_test)
print(classification_report(y_test, y_predicted))

dt = DecisionTree(random_state=0)
dt.fit(X_train, y_train)
y_predicted = dt.predict(X_test)
print(classification_report(y_test, y_predicted))
```
