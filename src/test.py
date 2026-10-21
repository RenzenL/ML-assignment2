import numpy as np
from sklearn.tree import DecisionTreeClassifier

X = np.array([[0], [1], [2], [3], [10], [11], [20], [21]])
y = np.array([0, 0, 0, 0, 1, 1, 2, 2])

clf = DecisionTreeClassifier(random_state=0).fit(X, y)

leaf_idx = clf.apply(X)
print(leaf_idx)                              # one number per sample

leaf_ids, counts = np.unique(leaf_idx, return_counts=True)
print(leaf_ids)                              # the distinct leaf numbers
print(counts)                                # how many samples sit in each

print(clf.get_n_leaves())                    # a single integer