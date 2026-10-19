import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_olivetti_faces
from common import flatten,mask, loader
from sklearn.tree import DecisionTreeClassifier,plot_tree

id_training, id_validation, id_testing = loader()
facesData=fetch_olivetti_faces()
#features
training=facesData['data'][id_training]
validation=facesData['data'][id_validation]
testing=facesData['data'][id_testing]

#labels
y_training=facesData['target'][id_training]
y_validation=facesData['target'][id_validation]
y_testing=facesData['target'][id_testing]

#size check
assert (training.shape==(240,4096))
assert (validation.shape==(80,4096))
assert (testing.shape==(80,4096))
assert (y_training.shape==(240,))
assert (y_validation.shape==(80,))
assert (y_testing.shape==(80,))

#q1(a)
results_list=[]
best_val_acc= -1.0
frozen_setting=None
frozen_tree=None

for depth in [2,4,8,16,None] :
    for leaf in [1,2,4] :
        clf=DecisionTreeClassifier(max_depth=depth,min_samples_leaf=leaf,random_state=0)
        clf.fit(training,y_training)

        leaf_idx=clf.apply(training)
        unqiue_leaves, counts=np.unique(leaf_idx,return_counts=True)
        n_leaves = clf.get_n_leaves()
        assert len(unqiue_leaves)==n_leaves

        mean_samples_per_leaf = np.mean(counts)
        frac_single_leaf= np.mean(counts == 1)
        val_acc=clf.score(validation,y_validation)
        test_acc=clf.score(testing, y_testing)

        results={"depth":depth,"leaf":leaf,
                 "val_acc":clf.score(validation,y_validation),
                 "train_acc":clf.score(training,y_training),
                 "leaf_count":clf.get_n_leaves(),
                 "mean_samples_per_leaf": mean_samples_per_leaf,
                 "frac_single_leaf": frac_single_leaf
                 }
        results_list.append(results)
        if val_acc>best_val_acc:
            best_val_acc=val_acc
            frozen_setting=(depth,leaf)
            frozen_tree=clf
        
        print(depth, leaf, results_list[-1]["val_acc"],
              results_list[-1]["train_acc"],
              results_list[-1]["leaf_count"]
              )