"""
Simulation study: certifying decisions from dependent AI judges.
n = 3 binary judges, binary truth Y. Joint distribution q over 16 cells (y, x).
Methods compared:
  MV     majority vote, no referral
  PLUG   empirical posterior per vote pattern from the audit, point thresholds
  CI     conditional-independence (naive Bayes) posterior fitted on the audit
  JS1    joint confidence set, audit cell constraints only (Clopper-Pearson, Bonferroni)
  JS2    JS1 + pooled audit events (prevalence, per-judge, pairwise) + unlabelled vote-pattern marginals
All population metrics are computed exactly from the true q.
"""
import itertools, json, sys
import numpy as np
from scipy.optimize import linprog
from scipy.stats import beta
from multiprocessing import Pool

N_J = 3
PATTERNS = list(itertools.product([0, 1], repeat=N_J))  # 8 vote patterns
CELLS = [(y, x) for y in (0, 1) for x in range(8)]        # index = y*8 + x
TAU_A, TAU_R = 0.05, 0.05
C_A, C_R, C_H = 10.0, 5.0, 1.0
ALPHA = 0.05
PREV = 0.4
N_UNLAB = 20000

def idx(y, x): return y * 8 + x

# ---------- data-generating regimes ----------
def q_from_correct_dist(pcorrect_given_y):
    """pcorrect_given_y[y][c] = P(correctness vector c | Y=y), c in {0,1}^3."""
    q = np.zeros(16)
    for y in (0, 1):
        py = PREV if y == 1 else 1 - PREV
        for x, pat in enumerate(PATTERNS):
            c = tuple(int(v == y) for v in pat)
            q[idx(y, x)] = py * pcorrect_given_y[y][c]
    return q

def regime_independent(p=0.85):
    d = {c: np.prod([p if ci else 1 - p for ci in c]) for c in PATTERNS}
    return q_from_correct_dist({0: d, 1: d})

def regime_copying(qC=0.8, a=0.5, p=0.9):
    d = {}
    for c in PATTERNS:
        tot = 0.0
        for shared_ok, ps in ((1, qC), (0, 1 - qC)):
            pr_ok = a * shared_ok + (1 - a) * p
            tot += ps * np.prod([pr_ok if ci else 1 - pr_ok for ci in c])
        d[c] = tot
    return q_from_correct_dist({0: d, 1: d})

def regime_higher_order(pk=(0.10, 0.03, 0.12, 0.75)):
    """Exchangeable: P(K=k) for k correct judges, uniform over which ones."""
    from math import comb
    d = {c: pk[sum(c)] / comb(3, sum(c)) for c in PATTERNS}
    return q_from_correct_dist({0: d, 1: d})

REGIMES = {"independent": regime_independent(),
           "shared_signal": regime_copying(),
           "higher_order": regime_higher_order()}

def true_post(q):
    num = np.array([q[idx(1, x)] for x in range(8)])
    den = np.array([q[idx(0, x)] + q[idx(1, x)] for x in range(8)])
    return num / den, den

# ---------- events for the confidence set ----------
def audit_events(pooled):
    ev = []
    for c in range(16):
        a = np.zeros(16); a[c] = 1; ev.append(("cell", a))
    if pooled:
        a = np.zeros(16); a[8:] = 1; ev.append(("prev", a))
        for y in (0, 1):
            for i in range(N_J):
                a = np.zeros(16)
                for x, pat in enumerate(PATTERNS):
                    if pat[i] == 1: a[idx(y, x)] = 1
                ev.append(("single", a))
            for i, j in itertools.combinations(range(N_J), 2):
                a = np.zeros(16)
                for x, pat in enumerate(PATTERNS):
                    if pat[i] == 1 and pat[j] == 1: a[idx(y, x)] = 1
                ev.append(("pair", a))
    return ev

def unlab_events():
    ev = []
    for x in range(8):
        a = np.zeros(16); a[idx(0, x)] = 1; a[idx(1, x)] = 1; ev.append(("marg", a))
    return ev

def cp(k, n, a):
    lo = 0.0 if k == 0 else beta.ppf(a / 2, k, n - k + 1)
    hi = 1.0 if k == n else beta.ppf(1 - a / 2, k + 1, n - k)
    return lo, hi

def build_set(audit_counts, m, unlab_counts, pooled):
    ev_a = audit_events(pooled)
    ev_u = unlab_events() if pooled else []
    K = len(ev_a) + len(ev_u)
    ak = ALPHA / K
    A, L, U = [], [], []
    for _, a in ev_a:
        lo, hi = cp(int(a @ audit_counts), m, ak); A.append(a); L.append(lo); U.append(hi)
    for _, a in ev_u:
        lo, hi = cp(int(a @ unlab_counts), N_UNLAB, ak); A.append(a); L.append(lo); U.append(hi)
    return np.array(A), np.array(L), np.array(U)

def bounds_for_z(A, L, U, x):
    """Sharp bounds on pi(x) over Q-hat via Charnes-Cooper. Returns (lo, hi) or None if unsupported/empty."""
    d = np.zeros(16); d[idx(0, x)] = 1; d[idx(1, x)] = 1
    # support check: min d^T q over Q-hat
    A_ub = np.vstack([A, -A]); b_ub = np.concatenate([U, -L])
    res = linprog(d, A_ub=A_ub, b_ub=b_ub, A_eq=np.ones((1, 16)), b_eq=[1], bounds=(0, None), method="highs")
    if res.status == 2: return "empty"
    if res.status != 0 or res.fun < 1e-9: return None
    # CC transform: variables [r(16), t]
    A_cc = np.vstack([np.hstack([A, -U[:, None]]), np.hstack([-A, L[:, None]])])
    b_cc = np.zeros(A_cc.shape[0])
    A_eq = np.vstack([np.concatenate([np.ones(16), [-1]]), np.concatenate([d, [0]])])
    b_eq = [0, 1]
    c = np.zeros(17); c[idx(1, x)] = 1
    lo = linprog(c, A_ub=A_cc, b_ub=b_cc, A_eq=A_eq, b_eq=b_eq, bounds=(0, None), method="highs")
    hi = linprog(-c, A_ub=A_cc, b_ub=b_cc, A_eq=A_eq, b_eq=b_eq, bounds=(0, None), method="highs")
    if lo.status != 0 or hi.status != 0: return None
    return lo.fun, -hi.fun

# ---------- decision rules ----------
def decide_interval(lo, hi):
    elig = {"H": C_H}
    if 1 - lo <= TAU_A: elig["A"] = C_A * (1 - lo)
    if hi <= TAU_R: elig["R"] = C_R * hi
    return min(elig, key=lambda k: (elig[k], "ARH".index(k)))

def evaluate(policy, q):
    """policy: list of 8 actions. Exact population metrics."""
    post, den = true_post(q)
    pA = pR = pH = fa = fr = loss = 0.0; viol = False
    for x in range(8):
        a = policy[x]; q0, q1 = q[idx(0, x)], q[idx(1, x)]
        if a == "A":
            pA += den[x]; fa += q0; loss += C_A * q0
            if den[x] > 0 and 1 - post[x] > TAU_A: viol = True
        elif a == "R":
            pR += den[x]; fr += q1; loss += C_R * q1
            if den[x] > 0 and post[x] > TAU_R: viol = True
        else:
            pH += den[x]; loss += C_H * den[x]
    return dict(pA=pA, pR=pR, pH=pH,
                fa_rate=fa / pA if pA > 0 else np.nan,
                fr_rate=fr / pR if pR > 0 else np.nan,
                loss=loss, viol=int(viol), accept111=int(policy[7] == "A"))

def one_rep(args):
    regime, m, seed = args
    q = REGIMES[regime]; rng = np.random.default_rng(seed)
    audit = rng.multinomial(m, q).astype(float)
    marg = np.array([q[idx(0, x)] + q[idx(1, x)] for x in range(8)])
    unl = np.zeros(16); unl[:8] = rng.multinomial(N_UNLAB, marg)  # stored in y=0 slots; events sum both
    out = {}
    # MV
    out["MV"] = evaluate(["A" if sum(p) >= 2 else "R" for p in PATTERNS], q)
    # PLUG
    pol = []
    for x in range(8):
        n0, n1 = audit[idx(0, x)], audit[idx(1, x)]
        if n0 + n1 == 0: pol.append("H"); continue
        p = n1 / (n0 + n1); pol.append(decide_interval(p, p))
    out["PLUG"] = evaluate(pol, q)
    # CI naive Bayes (Laplace smoothing)
    ny1 = audit[8:].sum(); ny0 = audit[:8].sum()
    prev = (ny1 + 1) / (m + 2)
    rates = {}
    for y in (0, 1):
        tot = audit[y * 8:(y + 1) * 8].sum()
        for i in range(N_J):
            k = sum(audit[idx(y, x)] for x, pat in enumerate(PATTERNS) if pat[i] == 1)
            rates[(y, i)] = (k + 1) / (tot + 2)
    pol = []
    for x, pat in enumerate(PATTERNS):
        l1 = prev * np.prod([rates[(1, i)] if pat[i] else 1 - rates[(1, i)] for i in range(N_J)])
        l0 = (1 - prev) * np.prod([rates[(0, i)] if pat[i] else 1 - rates[(0, i)] for i in range(N_J)])
        p = l1 / (l0 + l1); pol.append(decide_interval(p, p))
    out["CI"] = evaluate(pol, q)
    # JS1, JS2
    for name, pooled in (("JS1", False), ("JS2", True)):
        A, L, U = build_set(audit, m, unl, pooled)
        pol = []; empty = False; widths = []
        for x in range(8):
            b = bounds_for_z(A, L, U, x)
            if b == "empty": empty = True; pol.append("H"); continue
            if b is None: pol.append("H"); widths.append(1.0); continue
            pol.append(decide_interval(*b)); widths.append(b[1] - b[0])
        r = evaluate(pol, q); r["empty"] = int(empty); r["mean_width"] = float(np.mean(widths)) if widths else 1.0
        # width at the unanimous-yes pattern
        b = bounds_for_z(A, L, U, 7)
        r["width111"] = 1.0 if b in (None, "empty") else b[1] - b[0]
        out[name] = r
    return regime, m, out

if __name__ == "__main__":
    reps = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    ms = [250, 500, 1000, 2000, 5000, 10000, 20000]
    jobs = [(r, m, 1000003 * hash((r, m)) % (2**31) + k) for r in REGIMES for m in ms for k in range(reps)]
    # deterministic seeds
    jobs = [(r, m, i) for i, (r, m, _) in enumerate(jobs)]
    with Pool(2) as pool:
        res = pool.map(one_rep, jobs, chunksize=20)
    agg = {}
    for regime, m, out in res:
        for meth, r in out.items():
            agg.setdefault((regime, m, meth), []).append(r)
    summary = []
    for (regime, m, meth), rs in agg.items():
        row = dict(regime=regime, m=m, method=meth, reps=len(rs))
        for k in rs[0]:
            vals = np.array([r[k] for r in rs], dtype=float)
            row[k] = float(np.nanmean(vals))
            if k == "viol":
                p = row[k]; row["viol_se"] = float(np.sqrt(p * (1 - p) / len(rs)))
        summary.append(row)
    truth = {}
    for name, q in REGIMES.items():
        post, den = true_post(q)
        truth[name] = dict(post=[float(v) for v in post], marg=[float(v) for v in den],
                           acc=float(sum(q[idx(y, x)] for y in (0, 1) for x, p in enumerate(PATTERNS) for i in range(3) if p[i] == y) / 3))
    json.dump(dict(summary=summary, truth=truth), open("results.json", "w"), indent=1)
    print("done", len(res))
