import json
import os
import time

import joblib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from sklearn.datasets import fetch_olivetti_faces

from common import flatten, loader, mask
from ensembles import (SAMME, block_pixel_ids, fit_bagging, path_uses_block,
                       votes_of, weighted_votes)

SEED, K = 26007, 40
os.makedirs("logs", exist_ok=True)
os.makedirs("figures", exist_ok=True)
os.makedirs("models", exist_ok=True)


def dump(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else float(o))


# ----------------------------------------------------------------------------
# data and frozen Q1 models
# ----------------------------------------------------------------------------
id_tr, id_va, id_te = loader()
faces = fetch_olivetti_faces()
Xtr, ytr = faces.data[id_tr], faces.target[id_tr]
Xva, yva = faces.data[id_va], faces.target[id_va]
Xte, yte = faces.data[id_te], faces.target[id_te]
Xte_m = flatten(mask(faces.images[id_te]))                       # masked test set
assert Xte_m.shape == (80, 4096)
assert np.all(Xte_m.reshape(-1, 64, 64)[:, 24:40, 24:40] == 0)

recipe = json.load(open("logs/frozen_recipe.json"))
FD, FL = recipe["tree"]["max_depth"], recipe["tree"]["min_samples_leaf"]
B_FINAL = recipe["bagging"]["B"]
frozen_tree = joblib.load("models/frozen_tree.joblib")
final_bag = joblib.load("models/final_bagging.joblib")
assert len(final_bag) == B_FINAL
# the saved ensemble must be reproducible from SEED (catches a wrong seed in the recipe)
_chk = fit_bagging(Xtr, ytr, B_FINAL, FD, FL, SEED)
assert np.array_equal(votes_of(_chk, Xva).argmax(1), votes_of(final_bag, Xva).argmax(1)), \
    "refit bagging differs from models/final_bagging.joblib (check seed / recipe)"
print("frozen tree setting:", (FD, FL), "| final bagging B =", B_FINAL)

# ============================================================================
# Q2(a)  three-class toy example, one complete SAMME round logged
# ============================================================================
print("\n=== Q2(a) toy example ===")
Xt = np.array([[0.10, 0.20], [0.20, 0.10], [0.35, 0.60],      # class 0
               [0.50, 0.50], [0.60, 0.40], [0.45, 0.10],      # class 1
               [0.80, 0.90], [0.90, 0.70], [0.55, 0.30]])     # class 2
yt = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
K3 = 3
toy = SAMME(n_classes=K3, max_depth=1, n_rounds=3, seed=0).fit(Xt, yt)

w0, pred0, miss0 = toy.W_[0], toy.pred_[0], toy.miss_[0]
eps0, alpha0 = toy.eps_[0], toy.alphas_[0]
stump = toy.learners_[0].tree_
# independent recomputation of the round from the formulas
eps_chk = w0[miss0].sum()
alpha_chk = np.log((1 - eps_chk) / eps_chk) + np.log(K3 - 1)
w1 = w0 * np.exp(alpha_chk * miss0)
w1 = w1 / w1.sum()
assert np.isclose(eps0, eps_chk) and np.isclose(alpha0, alpha_chk)
if len(toy.W_) > 1 and not toy.events_:
    assert np.allclose(w1, toy.W_[1])             # same update the class used for round 2
# SAMME invariant: after the update the misclassified samples carry exactly (K-1)/K of the weight
assert np.isclose(w1[miss0].sum(), 1 - 1 / K3)

np.set_printoptions(precision=4, suppress=True)
print("stump: split on feature", int(stump.feature[0]), "at threshold", round(float(stump.threshold[0]), 3))
print("sample      :", np.arange(9))
print("true class  :", yt)
print("init weights:", w0)
print("predictions :", pred0)
print("misclassified:", miss0.astype(int))
print(f"weighted error eps = {eps0:.4f}")
print(f"alpha = log((1-eps)/eps) + log(K-1) = {np.log((1 - eps0) / eps0):.4f} + {np.log(K3 - 1):.4f} = {alpha0:.4f}")
print("updated weights (normalised):", w1)
print("weight on misclassified after update:", round(float(w1[miss0].sum()), 4), "= (K-1)/K =", round(1 - 1 / K3, 4))

# explicit handling of a degenerate weak learner: identical features => stump cannot beat random
try:
    SAMME(n_classes=3, max_depth=1, n_rounds=3).fit(np.zeros((9, 2)), yt)
except RuntimeError as e:
    print("degenerate weak learner handled explicitly ->", e)

dump({"X": Xt, "y": yt, "stump_feature": int(stump.feature[0]), "stump_threshold": float(stump.threshold[0]),
      "initial_weights": w0, "predictions": pred0, "misclassified": miss0, "weighted_error": eps0,
      "alpha": alpha0, "updated_weights": w1, "events": toy.events_}, "logs/q2a_toy.json")

# ============================================================================
# Q2(b)  grid over weak-learner depth and rounds, selected on clean validation accuracy
# ============================================================================
print("\n=== Q2(b) grid ===")
DEPTHS, ROUNDS = [1, 2, 3, 4], [10, 25, 50, 100]
grid, best = [], (-1.0, None)
for d in DEPTHS:
    m = SAMME(K, d, max(ROUNDS)).fit(Xtr, ytr)                 # one 100-round fit; first T learners = T-round fit
    P_tr, P_va = m.pred_, m.learner_predictions(Xva)
    for T in ROUNDS:
        Te = min(T, len(m.learners_))
        tr = (weighted_votes(P_tr, m.alphas_, Te, K).argmax(1) == ytr).mean()
        va = (weighted_votes(P_va, m.alphas_, Te, K).argmax(1) == yva).mean()
        grid.append({"depth": d, "rounds": T, "kept": Te, "train_acc": tr, "val_acc": va,
                     "n_events": len(m.events_)})
        print(f"depth={d} rounds={T:3d} kept={Te:3d} train={tr:.4f} val={va:.4f} events={len(m.events_)}")
        if va > best[0]:                                       # strict >: ties keep smaller depth, then fewer rounds
            best = (va, (d, T))
D_SEL, T_SEL = best[1]
print("selected (depth, rounds):", (D_SEL, T_SEL), "| val acc:", best[0])
dump(grid, "logs/q2_grid.json")

boost = SAMME(K, D_SEL, T_SEL).fit(Xtr, ytr)                   # the frozen boosting model
assert np.isclose((boost.predict(Xva) == yva).mean(), best[0])  # equals the grid entry (prefix property)
T_EFF = len(boost.learners_)
print("learners kept:", T_EFF, "| events:", boost.events_)
joblib.dump(boost, "models/frozen_boosting.joblib")
dump({"depth": D_SEL, "rounds": T_SEL, "kept": T_EFF, "seed": 0}, "logs/q2_config.json")

# diagnostics over rounds for the selected model
max_w = boost.W_.max(1)
top10 = np.sort(boost.W_, axis=1)[:, -10:].sum(1)
rr = np.arange(1, T_EFF + 1)
fig, ax = plt.subplots(1, 4, figsize=(14, 3.2))
ax[0].plot(rr, boost.eps_); ax[0].axhline(1 - 1 / K, ls=":", c="gray"); ax[0].set_title("weighted training error")
ax[1].plot(rr, boost.alphas_); ax[1].set_title("learner weight $\\alpha_t$")
ax[2].plot(rr, max_w); ax[2].set_yscale("log"); ax[2].set_title("max sample weight")
ax[3].plot(rr, top10); ax[3].set_title("weight on 10 heaviest images")
for a in ax:
    a.set_xlabel("round t"); a.grid(alpha=0.3)
plt.tight_layout(); plt.savefig("figures/q2b_rounds.pdf"); plt.close()
print("eps range:", boost.eps_.min().round(4), boost.eps_.max().round(4),
      "| alpha range:", boost.alphas_.min().round(3), boost.alphas_.max().round(3),
      "| max weight at end:", max_w[-1].round(4), "| top10 mass at end:", top10[-1].round(4))

# ============================================================================
# Q2(c)  boosting vs bagging at 25 / 50 / 100 learners, plus one heavy training face
# ============================================================================
print("\n=== Q2(c) boosting vs bagging ===")


def timed(fn, reps=3):
    ts, obj = [], None
    for _ in range(reps):
        t0 = time.perf_counter()
        obj = fn()
        ts.append(time.perf_counter() - t0)
    return obj, float(np.mean(ts)), float(np.std(ts))


cmp_rows = []
for L in [25, 50, 100]:
    bm, tb, sb = timed(lambda: SAMME(K, D_SEL, L).fit(Xtr, ytr))
    bg, tg, sg = timed(lambda: fit_bagging(Xtr, ytr, L, FD, FL, SEED))
    row_b = {"model": "boosting", "L": L, "kept": len(bm.learners_),
             "train": (bm.predict(Xtr) == ytr).mean(), "val": (bm.predict(Xva) == yva).mean(),
             "test": (bm.predict(Xte) == yte).mean(), "fit_s": tb, "fit_s_std": sb}
    row_g = {"model": "bagging", "L": L, "kept": L,
             "train": (votes_of(bg, Xtr).argmax(1) == ytr).mean(), "val": (votes_of(bg, Xva).argmax(1) == yva).mean(),
             "test": (votes_of(bg, Xte).argmax(1) == yte).mean(), "fit_s": tg, "fit_s_std": sg}
    for r in (row_b, row_g):
        cmp_rows.append(r)
        print(f"{r['model']:8s} L={L:3d} train={r['train']:.4f} val={r['val']:.4f} test={r['test']:.4f} "
              f"fit={r['fit_s']:.2f}s +- {r['fit_s_std']:.2f}")
dump(cmp_rows, "logs/q2c_compare.json")

# --- one training face with a very large boosting weight -------------------
W = boost.W_                                                    # (T, N)
i_h = int(W.max(0).argmax())                                    # heaviest training image over all rounds
t_peak = int(W[:, i_h].argmax())
miss_h, pred_h = boost.miss_[:, i_h], boost.pred_[:, i_h]
S, stage = np.zeros(K), []
for t in range(T_EFF):
    S[pred_h[t]] += boost.alphas_[t]
    stage.append(int(S.argmax()))                               # ensemble prediction after t+1 learners
stage = np.array(stage)
ok_stage = stage == ytr[i_h]
first_ok = int(np.argmax(ok_stage)) + 1 if ok_stage.any() else None
heavy = {"train_index": i_h, "identity": int(ytr[i_h]), "peak_weight": float(W[t_peak, i_h]),
         "peak_round": t_peak + 1, "peak_vs_uniform": float(W[t_peak, i_h] * len(ytr)),
         "rounds_misclassified": int(miss_h.sum()), "rounds_total": T_EFF,
         "alpha_mass_missed": float(boost.alphas_[miss_h].sum()),
         "alpha_mass_total": float(boost.alphas_.sum()),
         "first_round_ensemble_correct": first_ok, "ensemble_correct_at_end": bool(ok_stage[-1]),
         "ten_heaviest_peak": [int(k) for k in np.argsort(-W.max(0))[:10]]}
print("heavy face:", heavy)
dump({**heavy, "weight_history": W[:, i_h], "learner_pred": pred_h, "missed": miss_h,
      "ensemble_pred_after_t": stage}, "logs/q2c_heavy_face.json")

same_id = np.where(ytr == ytr[i_h])[0]
others = [k for k in same_id if k != i_h]
fig = plt.figure(figsize=(9, 4.6))
gs = GridSpec(2, 6, figure=fig)
for c, k in enumerate([i_h] + others):
    a = fig.add_subplot(gs[0, c]); a.imshow(faces.images[id_tr[k]], cmap="gray"); a.axis("off")
    a.set_title("heavy" if k == i_h else "same id", fontsize=7)
a = fig.add_subplot(gs[1, :3])
a.plot(rr, W[:, i_h]); a.set_yscale("log")
a.scatter(rr[miss_h], W[miss_h, i_h], c="r", s=12, label="learner wrong")
a.set_xlabel("round t"); a.set_title("weight of heavy face", fontsize=8); a.legend(fontsize=7)
a = fig.add_subplot(gs[1, 3:])
a.step(rr, ok_stage.astype(int), where="post"); a.set_ylim(-0.1, 1.1)
a.set_xlabel("round t"); a.set_title("ensemble correct on it after t learners", fontsize=8)
plt.tight_layout(); plt.savefig("figures/q2c_heavy_face.pdf"); plt.close()

# ============================================================================
# Q2(d)  frozen tree vs frozen bagging vs frozen boosting, clean and masked test
# ============================================================================
print("\n=== Q2(d) clean vs masked ===")
B = B_FINAL
v_c, v_m = votes_of(final_bag, Xte), votes_of(final_bag, Xte_m)
q_c, q_m = boost.predict_proba(Xte), boost.predict_proba(Xte_m)
tree_c, tree_m = frozen_tree.predict(Xte), frozen_tree.predict(Xte_m)
bag_c, bag_m = v_c.argmax(1), v_m.argmax(1)
bst_c, bst_m = q_c.argmax(1), q_m.argmax(1)

table = {
    "tree":     {"clean_acc": (tree_c == yte).mean(), "masked_acc": (tree_m == yte).mean(),
                 "clean_conf": frozen_tree.predict_proba(Xte).max(1).mean(),
                 "masked_conf": frozen_tree.predict_proba(Xte_m).max(1).mean()},
    "bagging":  {"clean_acc": (bag_c == yte).mean(), "masked_acc": (bag_m == yte).mean(),
                 "clean_conf": (v_c.max(1) / B).mean(), "masked_conf": (v_m.max(1) / B).mean()},
    "boosting": {"clean_acc": (bst_c == yte).mean(), "masked_acc": (bst_m == yte).mean(),
                 "clean_conf": q_c.max(1).mean(), "masked_conf": q_m.max(1).mean()},
}
for k, r in table.items():
    print(f"{k:9s} clean acc={r['clean_acc']:.4f} conf={r['clean_conf']:.4f} | "
          f"masked acc={r['masked_acc']:.4f} conf={r['masked_conf']:.4f}")

# exposure to the missing region: share of learners whose decision path for the CLEAN face tests a block pixel
blk = block_pixel_ids()
U_bag = np.array([path_uses_block(t, Xte, blk) for t in final_bag])             # (B, 80)
U_bst = np.array([path_uses_block(t, Xte, blk) for t in boost.learners_])       # (T, 80)
a_norm = boost.alphas_ / boost.alphas_.sum()
expo_bag, expo_bst = U_bag.mean(0), (a_norm[:, None] * U_bst).sum(0)
print("mean exposure to masked block (clean test): bagging", expo_bag.mean().round(3),
      "| boosting (alpha-weighted)", expo_bst.mean().round(3))

# per-learner predictions under both conditions
Pb_c = np.array([t.predict(Xte) for t in final_bag]).astype(int)
Pb_m = np.array([t.predict(Xte_m) for t in final_bag]).astype(int)
Pa_c, Pa_m = boost.learner_predictions(Xte), boost.learner_predictions(Xte_m)


def disagreement(vrow):                                          # fraction of tree pairs that disagree
    return float(1 - (vrow * (vrow - 1) / 2).sum() / (B * (B - 1) / 2))


# select the face by a fixed rule: first index where one ensemble stays correct and the other breaks
caseA = np.where((bag_c == yte) & (bag_m == yte) & (bst_c == yte) & (bst_m != yte))[0]   # bagging robust
caseB = np.where((bst_c == yte) & (bst_m == yte) & (bag_c == yte) & (bag_m != yte))[0]   # boosting robust
print("faces where bagging stays right but boosting breaks:", caseA.tolist())
print("faces where boosting stays right but bagging breaks:", caseB.tolist())
idx = np.arange(80)
d_share = (v_m[idx, yte] / B - v_c[idx, yte] / B) - (q_m[idx, yte] - q_c[idx, yte])
if len(caseA):
    i, why = int(caseA[0]), "bagging stays correct, boosting breaks"
elif len(caseB):
    i, why = int(caseB[0]), "boosting stays correct, bagging breaks"
else:
    i, why = int(np.abs(d_share).argmax()), "largest difference in change of true-class share"

flip = Pa_c[:, i] != Pa_m[:, i]                                  # boosting learners whose vote changes on this face
id_mass = np.array([boost.W_[t][ytr == yte[i]].sum() for t in range(T_EFF)])   # weight carried by the true identity
face = {
    "test_index": i, "selection_rule": why, "true": int(yte[i]),
    "bagging": {"clean_pred": int(bag_c[i]), "masked_pred": int(bag_m[i]),
                "true_votes_clean": int(v_c[i, yte[i]]), "true_votes_masked": int(v_m[i, yte[i]]),
                "winner_share_clean": float(v_c[i].max() / B), "winner_share_masked": float(v_m[i].max() / B),
                "tree_disagreement_clean": disagreement(v_c[i]), "tree_disagreement_masked": disagreement(v_m[i]),
                "trees_changing_vote": int((Pb_c[:, i] != Pb_m[:, i]).sum()),
                "trees_using_block_on_clean": float(U_bag[:, i].mean())},
    "boosting": {"clean_pred": int(bst_c[i]), "masked_pred": int(bst_m[i]),
                 "q_true_clean": float(q_c[i, yte[i]]), "q_true_masked": float(q_m[i, yte[i]]),
                 "winner_q_clean": float(q_c[i].max()), "winner_q_masked": float(q_m[i].max()),
                 "learners_changing_vote": int(flip.sum()), "alpha_mass_changing": float(a_norm[flip].sum()),
                 "alpha_mass_using_block_on_clean": float(expo_bst[i]),
                 "mean_true_identity_weight_changing": float(id_mass[flip].mean()) if flip.any() else None,
                 "mean_true_identity_weight_others": float(id_mass[~flip].mean()) if (~flip).any() else None,
                 "uniform_identity_weight": 1.0 / K,
                 "top_changing_learners": [
                     {"round": int(t + 1), "alpha": float(boost.alphas_[t]),
                      "top10_weight_mass": float(np.sort(boost.W_[t])[-10:].sum())}
                     for t in np.argsort(-boost.alphas_ * flip)[:5] if flip[t]]},
}
print(json.dumps(face, indent=2, default=float))
dump({"table": table, "mean_exposure": {"bagging": expo_bag.mean(), "boosting": expo_bst.mean()},
      "face": face}, "logs/q2d.json")

# figure: the face (clean/masked) and the two vote distributions
top_c = np.argsort(-(v_c[i] / B + q_c[i]))[:5]
cls = sorted(set(top_c.tolist() + [int(yte[i]), int(bag_m[i]), int(bst_m[i])]))
fig, ax = plt.subplots(1, 4, figsize=(13, 3.2))
ax[0].imshow(faces.images[id_te[i]], cmap="gray"); ax[0].set_title(f"clean (true id {yte[i]})", fontsize=8)
ax[1].imshow(flatten(mask(faces.images[id_te[i:i + 1]])).reshape(64, 64), cmap="gray"); ax[1].set_title("masked", fontsize=8)
for a in ax[:2]:
    a.axis("off")
x = np.arange(len(cls)); wd = 0.38
ax[2].bar(x - wd / 2, v_c[i, cls] / B, wd, label="clean"); ax[2].bar(x + wd / 2, v_m[i, cls] / B, wd, label="masked")
ax[2].set_title("bagging vote fraction", fontsize=8)
ax[3].bar(x - wd / 2, q_c[i, cls], wd, label="clean"); ax[3].bar(x + wd / 2, q_m[i, cls], wd, label="masked")
ax[3].set_title("boosting normalised weighted vote $q_k$", fontsize=8)
for a in ax[2:]:
    a.set_xticks(x); a.set_xticklabels(cls, fontsize=7); a.set_xlabel("identity", fontsize=8); a.legend(fontsize=7)
plt.tight_layout(); plt.savefig("figures/q2d_face.pdf"); plt.close()
print("\nQ2 done. Logs in logs/q2*.json, figures in figures/q2*.pdf, model in models/frozen_boosting.joblib")