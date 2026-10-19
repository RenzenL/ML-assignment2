import numpy as np
from sklearn.tree import DecisionTreeClassifier

X = np.array([[0], [1], [2], [3], [10], [11]])
y = np.array([0, 0, 0, 0, 1, 1])

clf = DecisionTreeClassifier(random_state=0).fit(X, y)
print(clf.apply(X))