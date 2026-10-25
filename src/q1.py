import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_olivetti_faces
from common import flatten,mask, loader
from sklearn.tree import DecisionTreeClassifier,plot_tree
import os
import json, os, joblib

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

#freeze: single tree + final bagged model
os.makedirs("figures", exist_ok=True)
final_B = 100                                    # chosen on validation accuracy
final_trees = trees[:final_B]                    # same rng stream and random_state=b, so = the B=100 ensemble

def votes_of(tree_list, X, K=40):
    pred = np.array([t.predict(X) for t in tree_list])        # (B, n)
    v = np.zeros((X.shape[0], K), int)
    for row in pred:
        v[np.arange(X.shape[0]), row] += 1
    return v                                                   # (n, 40)

# verify before anything downstream: must equal the B=100 row of bag_results
check = (votes_of(final_trees, validation).argmax(1) == y_validation).mean()
print("frozen setting:", frozen_setting, "| final bagged val acc:", check)
assert np.isclose(check, [r["validation_acc"] for r in bag_results if r["B"] == final_B][0])

recipe = {
    "tree": {"max_depth": frozen_depth, "min_samples_leaf": frozen_leaf, "random_state": 0},
    "bagging": {"B": final_B, "bootstrap_seed": 26007, "tree_random_state": "b (tree index)"},
    "notes": "ties in grid: first in grid order; vote ties: lowest identity number",
}
with open("logs/frozen_recipe.json", "w") as f:
    json.dump(recipe, f, indent=2)
with open("logs/q1_bagging.json", "w") as f:
    json.dump(bag_results, f, indent=2, default=float)
with open("logs/q1_tree_grid.json", "w") as f:
    json.dump(results_list, f, indent=2, default=float)
joblib.dump(frozen_tree, "models/frozen_tree.joblib") if os.makedirs("models", exist_ok=True) is None else None
joblib.dump(final_trees, "models/final_bagging.joblib")

# ================= Q1(d) =================
# masked test set: mask() needs (N,64,64) images, then flatten
testing_masked = flatten(mask(facesData["images"][id_testing]))
assert testing_masked.shape == (80, 4096)
assert np.all(testing_masked.reshape(-1, 64, 64)[:, 24:40, 24:40] == 0)

# predictions
v_clean,  v_masked  = votes_of(final_trees, testing), votes_of(final_trees, testing_masked)
bag_clean, bag_masked = v_clean.argmax(1), v_masked.argmax(1)
tree_clean  = frozen_tree.predict(testing)
tree_masked = frozen_tree.predict(testing_masked)

# accuracy and confidence
# tree confidence = probability of the predicted class at its leaf
# bagging confidence = winning vote fraction
q1d_table = {
    "tree_clean":  {"acc": (tree_clean  == y_testing).mean(),
                    "conf": frozen_tree.predict_proba(testing).max(1).mean()},
    "tree_masked": {"acc": (tree_masked == y_testing).mean(),
                    "conf": frozen_tree.predict_proba(testing_masked).max(1).mean()},
    "bag_clean":   {"acc": (bag_clean  == y_testing).mean(),
                    "conf": (v_clean.max(1)  / final_B).mean()},
    "bag_masked":  {"acc": (bag_masked == y_testing).mean(),
                    "conf": (v_masked.max(1) / final_B).mean()},
}
for k, r in q1d_table.items():
    print(f"{k:12s} acc={r['acc']:.4f}  conf={r['conf']:.4f}")

# example faces on the clean test set; selection rule: first two indices of each group
tree_ok, bag_ok = tree_clean == y_testing, bag_clean == y_testing
corrected  = np.where(~tree_ok &  bag_ok)[0]      # tree wrong, bagging right
both_wrong = np.where(~tree_ok & ~bag_ok)[0]      # both wrong
print("n corrected:", len(corrected), "| n both wrong:", len(both_wrong))
chosen = {"corrected": corrected[:2].tolist(), "both_wrong": both_wrong[:2].tolist()}

def describe(i):
    top = np.argsort(v_clean[i])[::-1][:5]
    return {"test_index": int(i), "true": int(y_testing[i]),
            "tree_pred": int(tree_clean[i]), "tree_conf": float(frozen_tree.predict_proba(testing[i:i+1]).max()),
            "bag_pred": int(bag_clean[i]),
            "votes_top5": {int(k): int(v_clean[i, k]) for k in top},     # out of final_B votes
            "votes_true_class": int(v_clean[i, y_testing[i]]),
            "votes_tree_pred": int(v_clean[i, tree_clean[i]])}

cases = {g: [describe(i) for i in idx] for g, idx in chosen.items()}
for g, lst in cases.items():
    for c in lst:
        print(g, c)

# figure: test face | a training face of the true identity | tree's guess | bagging's guess
tr_imgs = facesData["images"][id_training]
ex = lambda c: tr_imgs[y_training == c][0]
rows = [(g, i) for g, idx in chosen.items() for i in idx]
fig, axes = plt.subplots(len(rows), 4, figsize=(7, 1.8 * len(rows)), squeeze=False)
for r, (g, i) in enumerate(rows):
    imgs = [facesData["images"][id_testing[i]], ex(y_testing[i]), ex(tree_clean[i]), ex(bag_clean[i])]
    titles = [f"{g}: test", f"true {y_testing[i]}", f"tree {tree_clean[i]}", f"bag {bag_clean[i]}"]
    for c in range(4):
        axes[r, c].imshow(imgs[c], cmap="gray")
        axes[r, c].set_title(titles[c], fontsize=7)
        axes[r, c].axis("off")
plt.tight_layout()
plt.savefig("figures/q1d_examples.pdf")
plt.close()

with open("logs/q1d.json", "w") as f:
    json.dump({"table": q1d_table, "cases": cases}, f, indent=2, default=float)

# Q1(a): frozen setting's row from the grid, plus the test accuracy that part (a) asks for
row = [g for g in results_list if g["depth"] == frozen_depth and g["leaf"] == frozen_leaf][0]
print("Q1(a) frozen row:", row, "| test acc:", q1d_table["tree_clean"]["acc"])
