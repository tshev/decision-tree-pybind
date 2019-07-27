import sklearn.datasets
import os
from sklearn.model_selection import train_test_split

ds = sklearn.datasets.load_iris()
x = ds.data
y = ds.target
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.5)

def save_dataset(fname, x, y):
    with open(fname, "w") as fp:
        fp.write(f"{x.shape[0]} {x.shape[1]}\n")
        fp.write(" ".join(str(x) for x in x.reshape(x.shape[0] * x.shape[1])) + "\n")
        fp.write(f"{y.shape[0]}\n")
        fp.write(" ".join(str(x) for x in y))
    

save_dataset(os.path.join("data", "iris.train.sample.txt", X_train, y_train)
save_dataset(os.path.join("data", "iris.test.sample.txt", X_test, y_test)

