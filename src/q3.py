"""q3.py -- Question 3: bias and variance as measured quantities.

Run from the repository root:  python src/q3.py [--reps 30] [--jobs -1] [--recompute]

Needs the outputs of q1.py and q2.py: logs/frozen_recipe.json, models/frozen_tree.joblib,
models/final_bagging.joblib, models/frozen_boosting.joblib. Nothing is tuned here.
The expensive part (refitting everything on 30 resampled training sets) is cached in
logs/q3_raw.joblib; delete it or pass --recompute to refit.
"""
import argparse
import json
import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed
from sklearn.datasets import fetch_olivetti_faces
from sklearn.tree import DecisionTreeClassifier

from common import flatten, loader, mask
from ensembles import SAMME, fit_bagging, votes_of

SEED, K = 26007, 40
os.makedirs("logs", exist_ok=True)
os.makedirs("figures", exist_ok=True)


def dump(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else float(o))


# ----------------------------------------------------------------------------
# helpers (module level so joblib workers can use them)
# ----------------------------------------------------------------------------
def resample_ids(rng, y):
    """Within each identity draw 6 faces with replacement from that identity's 6 training faces.
    Returns positions into the ORIGINAL training array (length 240)."""
    out = []
    for c in range(K):
        pos = np.where(y == c)[0]
        out.append(rng.choice(pos, size=len(pos), replace=True))
    return np.concatenate(out)


def leaf_stats(tree, X, src):
    """Leaf occupancy counted in DISTINCT source images (duplicates from resampling count once).
    -> (n_leaves, mean images per leaf, fraction of leaves holding exactly one image)."""
    leaf = tree.apply(X)
    occ = np.array([len(np.unique(src[leaf == l])) for l in np.unique(leaf)])
    return len(occ), float(occ.mean()), float((occ == 1).mean())


def fit_bag_idx(X, y, src, B, max_depth, min_samples_leaf, seed):
    """Same trees as ensembles.fit_bagging (same rng stream, random_state=b), plus leaf statistics
    measured on each tree's own bootstrap sample."""
    N = len(y)
    rng = np.random.default_rng(seed)
    trees, stats = [], []
    for b in range(B):
        idx = rng.integers(0, N, size=N)
        t = DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=min_samples_leaf, random_state=b)
        t.fit(X[idx], y[idx])
        trees.append(t)
        nl, occ, sing = leaf_stats(t, X[idx], src[idx])
        stats.append((t.get_depth(), nl, occ, sing))
    return trees, np.array(stats)


def vote_matrix(preds, K=K):
    n = preds.shape[1]
    v = np.zeros((n, K))
    idx = np.arange(n)
    for row in preds:
        v[idx, row] += 1
    return v


def run_rep(r, seed_seq, Xtr, ytr, Xte, Xte_m, yte, cfgs, B, D_SEL, T_SEL):
    """Refit the three kinds of model on one resampled training set. No tuning."""
    rng = np.random.default_rng(seed_seq)
    src = resample_ids(rng, ytr)
    Xr, yr = Xtr[src], ytr[src]
    bag_seed = SEED + 1000 * (r + 1)                     # same bootstrap stream for every depth config
    out = {}
    for name, (d, l) in cfgs.items():
        # --- single tree: class vector = leaf class-probability vector ---
        t = DecisionTreeClassifier(max_depth=d, min_samples_leaf=l, random_state=0).fit(Xr, yr)
        P = t.predict_proba(Xte)
        assert P.shape[1] == K
        nl, occ, sing = leaf_stats(t, Xr, src)
        out["tree_" + name] = {"vec": P, "hard": P.argmax(1),
                               "acc_c": float((P.argmax(1) == yte).mean()),
                               "acc_m": float((t.predict(Xte_m) == yte).mean()),
                               "depth": t.get_depth(), "leaves": nl, "occ": occ, "single": sing}
        # --- bagging with B trees: class vector = vote fractions ---
        trees, st = fit_bag_idx(Xr, yr, src, B, d, l, bag_seed)
        pr = np.array([tr.predict(Xte) for tr in trees]).astype(int)
        pm = np.array([tr.predict(Xte_m) for tr in trees]).astype(int)
        v, vm = vote_matrix(pr), vote_matrix(pm)
        dis = 1.0 - ((v * (v - 1) / 2).sum(1) / (B * (B - 1) / 2)).mean()    # mean pairwise disagreement
        out["bag_" + name] = {"vec": v / B,
                              "acc_c": float((v.argmax(1) == yte).mean()),
                              "acc_m": float((vm.argmax(1) == yte).mean()),
                              "indiv_acc": float((pr == yte[None, :]).mean()),
                              "disagreement": float(dis), "depth": float(st[:, 0].mean()),
                              "leaves": float(st[:, 1].mean()), "occ": float(st[:, 2].mean()),
                              "single": float(st[:, 3].mean())}
    # --- boosting: class vector = normalised weighted vote q ---
    m = SAMME(K, D_SEL, T_SEL, seed=0).fit(Xr, yr)
    q = m.predict_proba(Xte)
    out["boost"] = {"vec": q, "acc_c": float((q.argmax(1) == yte).mean()),
                    "acc_m": float((m.predict(Xte_m) == yte).mean()),
                    "kept": len(m.learners_), "events": len(m.events_)}
    return out


def bias_var(V, y):
    """V: (R, N, K) class vectors. Bias^2 = mean_i ||pbar_i - e_i||^2,  Var = mean_{i,r} ||p_r_i - pbar_i||^2."""
    E = np.eye(K)[y]
    pbar = V.mean(0)
    bias2 = float(((pbar - E) ** 2).sum(1).mean())
    var = float(((V - pbar) ** 2).sum(2).mean())
    total = float(((V - E) ** 2).sum(2).mean())
    assert np.isclose(total, bias2 + var)               # expected squared error = bias^2 + variance exactly
    return bias2, var


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=30)
    ap.add_argument("--jobs", type=int, default=-1)
    ap.add_argument("--recompute", action="store_true")
    args = ap.parse_args()
    R = args.reps

    # ------------------------------------------------------------------ data, frozen models
    id_tr, id_va, id_te = loader()
    faces = fetch_olivetti_faces()
    Xtr, ytr = faces.data[id_tr], faces.target[id_tr]
    Xva, yva = faces.data[id_va], faces.target[id_va]
    Xte, yte = faces.data[id_te], faces.target[id_te]
    Xte_m = flatten(mask(faces.images[id_te]))
    assert Xte_m.shape == (80, 4096)

    recipe = json.load(open("logs/frozen_recipe.json"))
    FD, FL = recipe["tree"]["max_depth"], recipe["tree"]["min_samples_leaf"]
    B = recipe["bagging"]["B"]
    frozen_tree = joblib.load("models/frozen_tree.joblib")
    final_bag = joblib.load("models/final_bagging.joblib")
    boost0 = joblib.load("models/frozen_boosting.joblib")
    D_SEL, T_SEL = boost0.max_depth, boost0.n_rounds
    print(f"frozen: tree (depth={FD}, leaf={FL}) | bagging B={B} | boosting depth={D_SEL}, rounds={T_SEL} | R={R}")

    # fit_bag_idx must reproduce ensembles.fit_bagging exactly
    a = fit_bagging(Xtr, ytr, 5, FD, FL, SEED)
    b_, _ = fit_bag_idx(Xtr, ytr, np.arange(len(ytr)), 5, FD, FL, SEED)
    assert all(np.array_equal(x.predict(Xte), y.predict(Xte)) for x, y in zip(a, b_))

    # Depth configurations for Q3(b). The "selected" one is exactly the frozen tree / final bagging.
    # If the selected depth is already unbounded (None), a deeper tree is only possible by also
    # dropping the leaf-size limit, so "deep" = unbounded depth with min_samples_leaf=1.
    SHALLOW = 4 if (FD is None or FD > 4) else max(1, FD // 2)
    cfgs = {"shallow": (SHALLOW, FL), "selected": (FD, FL),
            "deep": (None, 1) if FD is None else (None, FL)}
    meta = {"R": R, "cfgs": {k: list(v) for k, v in cfgs.items()}, "B": B, "D": D_SEL, "T": T_SEL, "seed": SEED}
    print("depth configurations:", meta["cfgs"])

    # ------------------------------------------------------------------ 30 refits (cached)
    cache = "logs/q3_raw.joblib"
    outs = None
    if os.path.exists(cache) and not args.recompute:
        raw = joblib.load(cache)
        if raw["meta"] == meta:
            outs = raw["outs"]
            print("loaded cached refits from", cache)
    if outs is None:
        seeds = np.random.SeedSequence(SEED).spawn(R)
        outs = Parallel(n_jobs=args.jobs, verbose=5)(
            delayed(run_rep)(r, seeds[r], Xtr, ytr, Xte, Xte_m, yte, cfgs, B, D_SEL, T_SEL) for r in range(R))
        joblib.dump({"meta": meta, "outs": outs}, cache)

    keys = ["tree_" + n for n in cfgs] + ["bag_" + n for n in cfgs] + ["boost"]
    M = {}
    for k in keys:
        V = np.array([o[k]["vec"] for o in outs])                     # (R, 80, 40)
        b2, vr = bias_var(V, yte)
        M[k] = {"bias2": b2, "var": vr, "total": b2 + vr,
                "acc_c": np.array([o[k]["acc_c"] for o in outs]),
                "acc_m": np.array([o[k]["acc_m"] for o in outs])}
        for extra in ("depth", "leaves", "occ", "single", "indiv_acc", "disagreement", "kept", "events"):
            if extra in outs[0][k]:
                M[k][extra] = np.array([o[k][extra] for o in outs])
    # supplementary: single tree with hard one-hot class vectors instead of leaf probabilities
    onehot = {}
    for n in cfgs:
        H = np.array([o["tree_" + n]["hard"] for o in outs])
        onehot[n] = bias_var(np.eye(K)[H], yte)

    sd = lambda x: float(np.std(x, ddof=1))

    # ================================================================== Q3(a)
    print("\n=== Q3(a): selected models over", R, "refits (clean test set fixed) ===")
    sel = {"single tree": "tree_selected", "bagging": "bag_selected", "boosting": "boost"}
    print(f"{'model':12s} {'Bias^2':>8s} {'Var':>8s} {'Bias^2+Var':>10s} {'acc mean':>9s} {'acc sd':>8s}")
    for nm, k in sel.items():
        m = M[k]
        print(f"{nm:12s} {m['bias2']:8.4f} {m['var']:8.4f} {m['total']:10.4f} "
              f"{m['acc_c'].mean():9.4f} {sd(m['acc_c']):8.4f}")
    print("single tree, hard one-hot vectors instead of leaf probabilities: "
          f"Bias^2={onehot['selected'][0]:.4f} Var={onehot['selected'][1]:.4f}")

    # ================================================================== Q3(b)
    print("\n=== Q3(b): depth study (single tree vs 100-tree bagging, same resampled sets) ===")
    print(f"{'config':9s} {'depth':>6s} {'leaf':>4s} | {'tree B2':>8s} {'tree Var':>9s} | "
          f"{'bag B2':>8s} {'bag Var':>8s} | {'Var removed':>11s} {'% removed':>9s}")
    depth_rows = []
    for n, (d, l) in cfgs.items():
        t, g = M["tree_" + n], M["bag_" + n]
        rem = t["var"] - g["var"]
        depth_rows.append({"config": n, "max_depth": d, "min_samples_leaf": l,
                           "achieved_depth_tree": t["depth"].mean(), "tree_bias2": t["bias2"], "tree_var": t["var"],
                           "bag_bias2": g["bias2"], "bag_var": g["var"], "var_removed": rem,
                           "var_removed_pct": 100 * rem / t["var"]})
        print(f"{n:9s} {t['depth'].mean():6.1f} {l:4d} | {t['bias2']:8.4f} {t['var']:9.4f} | "
              f"{g['bias2']:8.4f} {g['var']:8.4f} | {rem:11.4f} {100 * rem / t['var']:8.1f}%")

    # ================================================================== Q3(c)
    print("\n=== Q3(c): is the deepest tree where bagging helps accuracy most? ===")
    print(f"{'config':9s} {'tree acc':>9s} {'bag acc':>8s} {'gain':>7s} {'SE':>6s} | {'leaves':>6s} {'imgs/leaf':>9s} "
          f"{'1-img leaves':>12s} | {'bagged-tree acc':>15s} {'disagree':>8s}")
    gain_rows = []
    for n in cfgs:
        t, g = M["tree_" + n], M["bag_" + n]
        gain = g["acc_c"] - t["acc_c"]                                    # paired: same resampled training sets
        gain_rows.append({"config": n, "tree_acc": t["acc_c"].mean(), "bag_acc": g["acc_c"].mean(),
                          "gain": gain.mean(), "gain_se": sd(gain) / np.sqrt(R),
                          "tree_leaves": t["leaves"].mean(), "tree_occupancy": t["occ"].mean(),
                          "tree_single_leaf_frac": t["single"].mean(),
                          "bagged_tree_occupancy": g["occ"].mean(), "bagged_tree_single_leaf_frac": g["single"].mean(),
                          "bagged_tree_indiv_acc": g["indiv_acc"].mean(), "bag_disagreement": g["disagreement"].mean()})
        print(f"{n:9s} {t['acc_c'].mean():9.4f} {g['acc_c'].mean():8.4f} {gain.mean():7.4f} {sd(gain) / np.sqrt(R):6.4f} | "
              f"{t['leaves'].mean():6.1f} {t['occ'].mean():9.2f} {t['single'].mean():12.3f} | "
              f"{g['indiv_acc'].mean():15.4f} {g['disagreement'].mean():8.4f}")
    best_gain = max(gain_rows, key=lambda r_: r_["gain"])["config"]
    best_rem = max(depth_rows, key=lambda r_: r_["var_removed"])["config"]
    deepest = max(depth_rows, key=lambda r_: r_["achieved_depth_tree"])["config"]
    print(f"largest accuracy gain from bagging: '{best_gain}' | most variance removed: '{best_rem}' | "
          f"deepest (by achieved depth): '{deepest}'")
    print("deepest tree is the one with the largest accuracy gain:", best_gain == deepest)

    # ================================================================== Q3(d)
    print("\n=== Q3(d): final table ===")
    frozen = {"single tree": ((frozen_tree.predict(Xte) == yte).mean(), (frozen_tree.predict(Xte_m) == yte).mean()),
              "bagging": ((votes_of(final_bag, Xte).argmax(1) == yte).mean(),
                          (votes_of(final_bag, Xte_m).argmax(1) == yte).mean()),
              "boosting": ((boost0.predict(Xte) == yte).mean(), (boost0.predict(Xte_m) == yte).mean())}
    final_rows = {}
    hdr = (f"{'model':12s} {'clean (frozen)':>14s} {'masked (frozen)':>15s} {'clean (refit mean)':>18s} "
           f"{'masked (refit mean)':>19s} {'drop':>6s} {'Bias^2':>8s} {'Var':>8s} {'acc sd':>7s}")
    print(hdr)
    for nm, k in sel.items():
        m = M[k]
        row = {"clean_frozen": frozen[nm][0], "masked_frozen": frozen[nm][1],
               "clean_refit_mean": m["acc_c"].mean(), "clean_refit_sd": sd(m["acc_c"]),
               "masked_refit_mean": m["acc_m"].mean(), "masked_refit_sd": sd(m["acc_m"]),
               "drop_refit": m["acc_c"].mean() - m["acc_m"].mean(), "bias2": m["bias2"], "var": m["var"]}
        final_rows[nm] = row
        print(f"{nm:12s} {row['clean_frozen']:14.4f} {row['masked_frozen']:15.4f} {row['clean_refit_mean']:18.4f} "
              f"{row['masked_refit_mean']:19.4f} {row['drop_refit']:6.4f} {row['bias2']:8.4f} {row['var']:8.4f} "
              f"{row['clean_refit_sd']:7.4f}")
    lowest_var = min(final_rows, key=lambda x: final_rows[x]["var"])
    best_acc = max(final_rows, key=lambda x: final_rows[x]["clean_refit_mean"])
    most_sens = max(final_rows, key=lambda x: final_rows[x]["drop_refit"])
    most_stable_acc = min(final_rows, key=lambda x: final_rows[x]["clean_refit_sd"])
    print(f"lowest variance: {lowest_var} | best mean clean accuracy: {best_acc} | "
          f"smallest accuracy sd: {most_stable_acc} | largest clean->masked drop: {most_sens}")
    print("lowest-variance model is the most accurate one:", lowest_var == best_acc)

    # ------------------------------------------------------------------ logs
    dump({"meta": meta, "frozen_models": {"tree": [FD, FL], "bagging_B": B, "boosting": [D_SEL, T_SEL]},
          "q3a": {nm: {"bias2": M[k]["bias2"], "var": M[k]["var"], "acc_mean": M[k]["acc_c"].mean(),
                       "acc_sd": sd(M[k]["acc_c"])} for nm, k in sel.items()},
          "tree_onehot_bias_var": {n: list(v) for n, v in onehot.items()},
          "q3b": depth_rows, "q3c": gain_rows, "q3d": final_rows,
          "boosting_kept_learners_mean": M["boost"]["kept"].mean(), "boosting_events_total": int(M["boost"]["events"].sum()),
          "per_refit_accuracy": {k: {"clean": M[k]["acc_c"], "masked": M[k]["acc_m"]} for k in keys}},
         "logs/q3.json")

    # ------------------------------------------------------------------ figures
    fig, ax = plt.subplots(1, 2, figsize=(8, 3.2))
    for a_, key, title in zip(ax, ["acc_c", "acc_m"], ["clean test", "masked test"]):
        a_.boxplot([M[k][key] for k in sel.values()])
        a_.set_xticks([1, 2, 3]); a_.set_xticklabels(list(sel.keys()), fontsize=8)
        a_.set_title(f"accuracy over {R} refits: {title}", fontsize=9); a_.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig("figures/q3a_accuracy_boxplots.pdf"); plt.close()

    labels = [f"{n}\n(depth~{M['tree_' + n]['depth'].mean():.0f})" for n in cfgs]
    x, wd = np.arange(len(cfgs)), 0.38
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
    for a_, key, title in zip(ax, ["bias2", "var"], ["Bias$^2$", "Variance"]):
        a_.bar(x - wd / 2, [M["tree_" + n][key] for n in cfgs], wd, label="single tree")
        a_.bar(x + wd / 2, [M["bag_" + n][key] for n in cfgs], wd, label="bagging (100)")
        a_.set_xticks(x); a_.set_xticklabels(labels, fontsize=8); a_.set_title(title, fontsize=9)
        a_.legend(fontsize=7); a_.grid(alpha=0.3, axis="y")
    plt.tight_layout(); plt.savefig("figures/q3b_depth_bias_variance.pdf"); plt.close()
    print("\nQ3 done: logs/q3.json, figures/q3a_*.pdf, figures/q3b_*.pdf")