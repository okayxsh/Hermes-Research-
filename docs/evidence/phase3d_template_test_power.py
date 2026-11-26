"""Planning simulation (synthetic; no data): task-level exact McNemar vs template-level sign-flip tests on the real
roster allocation (25 templates x 3 tasks + 2 x 2 = 79 tasks), one inference seed.
Tests:  T  = task-level sign flip (= exact McNemar),
        TM = template-level sign flip on template MEANS (template-weighted; current SAP sensitivity),
        TS = template-level sign flip on template SUMS (statistic |sum_g S_g| = |sum_i d_i| -> task-weighted).
Null distributions are exact (dynamic programming over integer-scaled values), so no Monte-Carlo error in p-values.
Data model: logit P(success) = logit(p0) + u_task + v_template + arm * (delta_logit + tau_template)
  u ~ N(0, sd_task) and v ~ N(0, sd_tpl) act on BOTH arms; tau ~ N(0, sd_het) is a template-correlated effect."""
import math
import random
import sys
from collections import defaultdict
from statistics import fmean

SIZES = [3] * 25 + [2] * 2


def expit(x):
    return 1 / (1 + math.exp(-x)) if x >= 0 else math.exp(x) / (1 + math.exp(x))


def logit(p):
    return math.log(p / (1 - p))


def signflip_p(values):
    """Exact two-sided p for |sum s_i v_i| >= |sum v_i| with integer v_i."""
    vals = [abs(v) for v in values if v != 0]
    obs = abs(sum(values))
    if not vals:
        return 1.0
    dist = defaultdict(float)
    dist[0] = 1.0
    for v in vals:
        nxt = defaultdict(float)
        for s, pr in dist.items():
            nxt[s + v] += pr / 2
            nxt[s - v] += pr / 2
        dist = nxt
    return sum(pr for s, pr in dist.items() if abs(s) >= obs)


def delta_for(p0, target):
    lo, hi = 0.0, 6.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if expit(logit(p0) + mid) - p0 < target:
            lo = mid
        else:
            hi = mid
    return lo


def run(p0, target_delta, sd_task, sd_tpl, sd_het, reps, seed):
    rng = random.Random(seed)
    dl = delta_for(p0, target_delta) if target_delta else 0.0
    rej = {"T": 0, "TM": 0, "TS": 0}
    eff = []
    for _ in range(reps):
        sums, means, tasks = [], [], []
        for n in SIZES:
            v, tau = rng.gauss(0, sd_tpl), rng.gauss(0, sd_het)
            ds = []
            for _ in range(n):
                u = rng.gauss(0, sd_task)
                base = logit(p0) + u + v
                y0 = rng.random() < expit(base)
                y1 = rng.random() < expit(base + dl + tau)
                ds.append(int(y1) - int(y0))
            tasks += ds
            sums.append(sum(ds))
            means.append(sum(ds) * (6 // n))   # template mean scaled by 6 -> integer
        eff.append(fmean(tasks))
        rej["T"] += signflip_p(tasks) < 0.05
        rej["TM"] += signflip_p(means) < 0.05
        rej["TS"] += signflip_p(sums) < 0.05
    return {"p0": p0, "delta": target_delta, "sd_task": sd_task, "sd_tpl": sd_tpl, "sd_het": sd_het, "reps": reps,
            "mean_task_weighted_effect": round(fmean(eff), 3), **{k: round(v / reps, 3) for k, v in rej.items()}}



def swap(h, reps, seed, sd_task=2.0, sd_tpl=1.0, p0=0.25):
    rng = random.Random(seed)
    rej = {"T": 0, "TM": 0, "TS": 0}
    eff = []
    for _ in range(reps):
        sums, means, tasks = [], [], []
        for n in SIZES:
            v, sg = rng.gauss(0, sd_tpl), rng.choice((1, -1))
            ds = []
            for _ in range(n):
                b = logit(p0) + rng.gauss(0, sd_task) + v
                ds.append(int(rng.random() < expit(b + sg * h)) - int(rng.random() < expit(b - sg * h)))
            tasks += ds; sums.append(sum(ds)); means.append(sum(ds) * (6 // n))
        eff.append(fmean(tasks))
        rej["T"] += signflip_p(tasks) < 0.05; rej["TM"] += signflip_p(means) < 0.05; rej["TS"] += signflip_p(sums) < 0.05
    return {"weak_null_random_sign_h": h, "reps": reps, "mean_effect": round(fmean(eff), 3), **{k: round(x / reps, 3) for k, x in rej.items()}}


if __name__ == "__main__":
    import sys
    reps = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
    # sharp null; weak null with template-correlated effects of random sign (mean effect exactly 0)
    print(run(0.25, 0.0, 2.0, 1.0, 0.0, reps, 11))
    for h in (0.5, 1.0, 1.5):
        print(swap(h, reps, 21))
    # power at realised task-weighted effects of about 11 and 20 points, without / with template heterogeneity
    for target in (0.18, 0.33):
        for het in (0.0, 1.0):
            print(run(0.25, target, 2.0, 1.0, het, reps, 23))
