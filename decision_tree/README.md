# Usage example

```python
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

# Micro benchmark
## Code
```python
from sklearn.datasets import load_iris
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from decision_tree_pybind import DecisionTree
import numpy as np
import timeit

iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.3, random_state=4)

dtc = DecisionTreeClassifier(random_state=0)
measurements = 100
print("fit time: %f" % (timeit.timeit(lambda: dtc.fit(X_train, y_train), number=measurements)))
print("predict time: %f" % (timeit.timeit(lambda: dtc.predict(X_test), number=measurements)))
y_predicted = dtc.predict(X_test)
print(classification_report(y_test, y_predicted))

dt = DecisionTree(random_state=0)
print("fit time: %f" % (timeit.timeit(lambda: dt.fit(X_train, y_train), number=measurements)))
print("predict time: %f" % (timeit.timeit(lambda: dt.predict(X_test), number=measurements)))
y_predicted = dt.predict(X_test)
print(classification_report(y_test, y_predicted))
```

## Results
```
fit time: 0.021076
predict time: 0.004165
              precision    recall  f1-score   support

           0       1.00      1.00      1.00        21
           1       1.00      0.90      0.95        10
           2       0.93      1.00      0.97        14

    accuracy                           0.98        45
   macro avg       0.98      0.97      0.97        45
weighted avg       0.98      0.98      0.98        45

fit time: 0.027699
predict time: 0.000281
              precision    recall  f1-score   support

           0       1.00      1.00      1.00        21
           1       0.91      1.00      0.95        10
           2       1.00      0.93      0.96        14

    accuracy                           0.98        45
   macro avg       0.97      0.98      0.97        45
weighted avg       0.98      0.98      0.98        45
```
