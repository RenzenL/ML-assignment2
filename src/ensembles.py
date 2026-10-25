
import numpy as np
from sklearn.tree import DecisionTreeClassifier


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
def weighted_votes(preds, alphas, T, K):
    """S_k(x) = sum_{t<T} alpha_t * 1[h_t(x) = k].   preds: (T_total, n) -> (n, K)."""
    n = preds.shape[1]
    S = np.zeros((n, K))
    idx = np.arange(n)
    for t in range(T):
        S[idx, preds[t]] += alphas[t]
    return S


def votes_of(trees, X, K=40):
    """Plurality-vote counts of a list of trees. -> (n, K) integer counts."""
    preds = np.array([t.predict(X) for t in trees]).astype(int)
    v = np.zeros((X.shape[0], K), int)
    idx = np.arange(X.shape[0])
    for row in preds:
        v[idx, row] += 1
    return v


def block_pixel_ids(lo=24, hi=40, width=64):
    """Flattened pixel indices of the centred block (rows/cols lo..hi-1)."""
    r = np.arange(lo, hi)
    return (r[:, None] * width + r[None, :]).ravel()


def path_uses_block(tree, X, block_ids):
    """True for every row of X whose root-to-leaf path tests a pixel inside the block."""
    in_block = np.isin(tree.tree_.feature, block_ids).astype(np.int32)  # leaves have feature -2
    D = tree.decision_path(X)                                           # sparse (n, n_nodes)
    return np.asarray(D.dot(in_block)).ravel() > 0


# ----------------------------------------------------------------------------
# bagging
# ----------------------------------------------------------------------------
def fit_bagging(X, y, B, max_depth, min_samples_leaf, seed=26007):
    """B trees, each on a bootstrap sample of the original size.
    Same rng stream and random_state=b as q1.py, so the first B trees of a larger
    ensemble are exactly the B-tree ensemble."""
    N = len(y)
    rng = np.random.default_rng(seed)
    trees = []
    for b in range(B):
        idx = rng.integers(0, N, size=N)
        t = DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=min_samples_leaf,
                                   random_state=b)
        t.fit(X[idx], y[idx])
        trees.append(t)
    return trees


# ----------------------------------------------------------------------------
# SAMME
# ----------------------------------------------------------------------------
class SAMME:
    """Multiclass AdaBoost (SAMME).

        eps_t   = sum_i w_i 1[h_t(x_i) != y_i]             (w normalised)
        alpha_t = log((1 - eps_t) / eps_t) + log(K - 1)
        w_i    <- w_i * exp(alpha_t * 1[h_t(x_i) != y_i]),  then renormalised
        S_k(x)  = sum_t alpha_t 1[h_t(x) = k],   q_k(x) = S_k(x) / sum_t alpha_t

    Explicit handling of degenerate weak learners
    ---------------------------------------------
    * eps_t >= 1 - 1/K  (not better than random multiclass guessing; alpha_t <= 0):
      the learner is DISCARDED (never added). If the weights are not uniform, they are
      reset to uniform and boosting continues with a new attempt; if they already were
      uniform (nothing left to reweight) boosting stops early. Every such event is
      recorded in `events_`. Attempts are capped at 3 * n_rounds.
    * eps_t == 0 (perfect weighted fit): eps is clipped to `eps_floor` (finite alpha),
      the learner is kept, and boosting stops, because the weights would not change and
      later learners would repeat this one.

    `n_rounds` counts KEPT learners. Because the algorithm only looks backwards, the
    first T learners of a 100-round fit are exactly a T-round fit.
    """

    def __init__(self, n_classes, max_depth, n_rounds, seed=0, eps_floor=1e-10):
        self.K = n_classes
        self.max_depth = max_depth
        self.n_rounds = n_rounds
        self.seed = seed
        self.eps_floor = eps_floor

    def fit(self, X, y):
        K = self.K
        assert np.array_equal(np.unique(y), np.arange(K)), "labels must be 0..K-1, all present"
        N = len(y)
        w = np.full(N, 1.0 / N)
        learners, alphas, W, EPS, MISS, PRED = [], [], [], [], [], []
        self.events_ = []
        attempts = 0
        while len(learners) < self.n_rounds and attempts < 3 * self.n_rounds:
            attempts += 1
            tree = DecisionTreeClassifier(max_depth=self.max_depth,
                                          random_state=self.seed + attempts)
            tree.fit(X, y, sample_weight=w)
            pred = tree.predict(X).astype(int)
            miss = pred != y
            eps = float(w[miss].sum() / w.sum())

            if eps >= 1.0 - 1.0 / K - 1e-9:                    # not better than random guessing
                at_uniform = (not learners) or np.allclose(w, 1.0 / N)
                self.events_.append({"attempt": attempts, "kept_so_far": len(learners),
                                     "eps": eps,
                                     "action": "discarded; stop" if at_uniform
                                               else "discarded; weights reset to uniform"})
                if at_uniform:
                    break
                w = np.full(N, 1.0 / N)
                continue

            stop = False
            eps_c = eps
            if eps <= self.eps_floor:                          # perfect weak learner
                eps_c, stop = self.eps_floor, True
                self.events_.append({"attempt": attempts, "kept_so_far": len(learners),
                                     "eps": eps, "action": "eps clipped to floor; stop"})
            alpha = np.log((1.0 - eps_c) / eps_c) + np.log(K - 1)

            learners.append(tree)
            alphas.append(alpha)
            W.append(w.copy())                                 # weights this learner was trained with
            EPS.append(eps)
            MISS.append(miss.copy())
            PRED.append(pred.copy())

            w = w * np.exp(alpha * miss)                       # SAMME update
            w = w / w.sum()
            if stop:
                break

        if not learners:
            raise RuntimeError("no weak learner better than random guessing")
        self.learners_ = learners
        self.alphas_ = np.array(alphas)
        self.W_ = np.array(W)            # (T, N): normalised weights used to train learner t
        self.eps_ = np.array(EPS)        # (T,)  weighted training error of learner t
        self.miss_ = np.array(MISS)      # (T, N) learner t misclassified training sample i
        self.pred_ = np.array(PRED)      # (T, N) learner t's prediction on training sample i
        self.final_weights_ = w          # weights after the last update
        return self

    # ---- prediction -------------------------------------------------------
    def learner_predictions(self, X):
        return np.array([t.predict(X) for t in self.learners_]).astype(int)    # (T, n)

    def decision_scores(self, X, T=None):
        T = len(self.learners_) if T is None else min(T, len(self.learners_))
        return weighted_votes(self.learner_predictions(X), self.alphas_, T, self.K)

    def predict_proba(self, X, T=None):
        """Normalised weighted vote q_k(x) = S_k(x) / sum_t alpha_t."""
        T = len(self.learners_) if T is None else min(T, len(self.learners_))
        return self.decision_scores(X, T) / self.alphas_[:T].sum()

    def predict(self, X, T=None):
        return self.decision_scores(X, T).argmax(1)