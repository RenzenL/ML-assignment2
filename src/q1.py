import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_olivetti_faces
from common import flatten,mask, loader
from sklearn.tree import DecisionTreeClassifier,plot_tree
import os

#making log file to store results
os.makedirs("logs", exist_ok=True)

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

        results={"depth":depth,"leaf":leaf,
                 "val_acc":val_acc,
                 "train_acc":clf.score(training,y_training),
                 "leaf_count":clf.get_n_leaves(),
                 "mean_samples_per_leaf": mean_samples_per_leaf,
                 "frac_single_leaf": frac_single_leaf
                 }
        results_list.append(results)
        #shallower depth first and smaller leaf size first
        if val_acc>best_val_acc:
            best_val_acc=val_acc
            frozen_setting=(depth,leaf)
            frozen_tree=clf
        
        print(depth, leaf, results_list[-1]["val_acc"],
              results_list[-1]["train_acc"],
              results_list[-1]["leaf_count"]
              )
#q1 (b & c)
frozen_depth,frozen_leaf=frozen_setting
N=len(training)
bag_results=[]
for B in [1,5,10,25,50,100,200]:
    oob_idx_trees=[]
    rng=np.random.default_rng(26007)
    trees=[]
    distinct_counts=[]
    boot_idx_tree=[]
    #training B separate trees:
    for b in range(B):
        oob_idx=np.ones(N, dtype=bool)
        boot_idx=rng.integers(0, N, size=N)
        oob_idx[boot_idx]=0
        oob_idx_trees.append(oob_idx)
        boot_idx_tree.append(boot_idx)
        distinct_counts.append(len(np.unique(boot_idx)))
        X_boot=training[boot_idx]
        y_boot=y_training[boot_idx]
        tree=DecisionTreeClassifier(max_depth=frozen_depth,
                                min_samples_leaf=frozen_leaf,
                                random_state=b)
        tree.fit(X_boot,y_boot)
        trees.append(tree)
    #calculate parameters for ensemble with B trees
    tree_pred_train=np.array([t.predict(training) for t in trees])
    tree_pred_val=np.array([t.predict(validation) for t in trees])
    tree_pred_test=np.array([t.predict(testing) for t in trees])
    #votes shape(240,40) :
    votes_train = np.array([np.bincount(tree_pred_train[:, j], minlength=40)
                  for j in range(tree_pred_train.shape[1])])
    votes_val = np.array([np.bincount(tree_pred_val[:, j], minlength=40)
                      for j in range(tree_pred_val.shape[1])])
    votes_test = np.array([np.bincount(tree_pred_test[:, j], minlength=40)
                      for j in range(tree_pred_test.shape[1])])
    
    
    oob_votes = np.zeros((N, 40), int)            # one row per training image
    for b in range(B):
        oob_pos = np.where(oob_idx_trees[b])[0]   # positions tree b never saw
        if len(oob_pos) == 0:
            continue
        p = trees[b].predict(training[oob_pos])   # predict only, no fitting
        oob_votes[oob_pos, p] += 1
    covered = oob_votes.sum(1) > 0                # images with at least one OOB vote        
        
    #winners_shape(240,)  contains the winning idenitiy per image :
    winners_train = votes_train.argmax(axis=1)
    winners_val = votes_val.argmax(axis=1)
    winners_test = votes_test.argmax(axis=1)
    

    accuracy_train=(winners_train==y_training).mean()
    accuracy_val=(winners_val==y_validation).mean()
    accuracy_testing=(winners_test==y_testing).mean()
    accuracy_oob = (oob_votes[covered].argmax(1) == y_training[covered]).mean()
    oob_coverage = covered.mean()
    
    mean_distinct_count=np.mean(distinct_counts)

    avg_pairwise_dis=0
    #calculate the disagreement :
    for a in range  (0,len(trees)):
        for b in range (a+1,len(trees)):
            avg_pairwise_dis+=(tree_pred_val[a]!=tree_pred_val[b]).mean()
    avg_pairwise_dis=avg_pairwise_dis/(B * (B - 1) / 2) if B > 1 else np.nan

    #mean vote margin : sort the votes in val set and diff btwn 1st and 2nd
    sorted_votes=np.sort(votes_val,axis=1)
    top1=sorted_votes[:,-1]
    top2=sorted_votes[:,-2]
    margin = (top1-top2)/B
    mean_margin=margin.mean()

    rec = {"B": B, "train_acc": accuracy_train,"validation_acc": accuracy_val,
           "testing_acc":accuracy_testing, "mean_distinct": mean_distinct_count,
           "oob_acc":accuracy_oob, "oob_coverage": oob_coverage,
           "avg_pairwise_dis":avg_pairwise_dis, "mean_margin": mean_margin}
    bag_results.append(rec)

#time to plot the oobAcc, avg_pairwise_dis and mean margin against B :
#extract each column from bag results
B_val_extracted = [r["B"] for r in bag_results]
oob_acc_extracted = [r["oob_acc"] for r in bag_results]
avg_pw_dis_extracted = [r["avg_pairwise_dis"] for r in bag_results]
mean_margin_extracted = [r["mean_margin"] for r in bag_results]

fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
axes[0].plot(B_val_extracted, oob_acc_extracted, marker="o")
axes[0].set_title("OOB accuracy")
axes[1].plot(B_val_extracted, avg_pw_dis_extracted, marker="o")
axes[1].set_title("Avg pairwise disagreement")
axes[2].plot(B_val_extracted, mean_margin_extracted, marker="o")
axes[2].set_title("Mean vote margin")
for ax in axes:
    ax.set_xlabel("B (number of trees)")
    ax.set_xscale("log")     
plt.tight_layout()                      
plt.savefig("q1c_curves.pdf")
plt.show()

for result in bag_results :
    print(result)