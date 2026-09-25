"""Runs every experiment of the Demo B report and stores results in results.pkl and numbers.tex."""
import numpy as np, pickle, time, json
from demoB import DemoB, run, seed_labels, aggregate

T0 = time.time()
RES = {}
NUM = {}
MARGIN = 0.03  # |log rho_pred| below this -> predicted boundary point


def region_of(lo, lc):
    return np.array([o + c if (o in ("I", "D") and c in ("I", "D")) else "--" for o, c in zip(lo, lc)])


def grid_experiment(demo, betas, kappas, R, T=80, seed=1, tag=""):
    B, K = np.meshgrid(betas, kappas, indexing="ij"); b, k = B.ravel(), K.ravel()
    rOL, rCL = demo.predicted_rho(b, k)
    out = dict(betas=betas, kappas=kappas, b=b, k=k, rOL_pred=rOL, rCL_pred=rCL)
    for loop, closed in (("OL", False), ("CL", True)):
        r = run(demo, b, k, closed, R=R, T=T, seed=seed)
        sl = seed_labels(r); ag = aggregate(sl)
        cons = np.array([np.mean(sl[i] == ag[i]) if ag[i] in ("I", "D") else np.nan for i in range(len(b))])
        out[loop] = dict(lab=ag, seedlab=sl, rho=np.median(r["rho"], 1), rho_q=np.quantile(r["rho"], [0.25, 0.75], axis=1),
                         TO=np.median(r["TO"], 1), TH=np.median(r["TH"], 1), cons=cons)
    reg = region_of(out["OL"]["lab"], out["CL"]["lab"])
    pred = np.char.add(np.where(rOL > 1, "D", "I"), np.where(rCL > 1, "D", "I"))
    near = (np.abs(np.log(rOL)) < MARGIN) | (np.abs(np.log(rCL)) < MARGIN)
    out.update(region=reg, pred=pred, near=near)
    s = dict(n=len(b), counts={q: int((reg == q).sum()) for q in ("II", "ID", "DI", "DD")},
             pred_counts={q: int((pred == q).sum()) for q in ("II", "ID", "DI", "DD")},
             nB=int(((out["OL"]["lab"] == "B") | (out["CL"]["lab"] == "B")).sum()),
             nX=int(((out["OL"]["lab"] == "X") | (out["CL"]["lab"] == "X")).sum()),
             agree_all=float((reg == pred).mean()), agree_far=float((reg == pred)[~near].mean()), n_far=int((~near).sum()),
             n_near=int(near.sum()),
             cons=float(np.nanmean(np.r_[out["OL"]["cons"], out["CL"]["cons"]])))
    rerr = np.r_[np.abs(np.log(out["OL"]["rho"]) - np.log(rOL))[~near], np.abs(np.log(out["CL"]["rho"]) - np.log(rCL))[~near]]
    s["rho_logerr_med"] = float(np.median(rerr)); s["rho_logerr_p95"] = float(np.quantile(rerr, 0.95))
    out["summary"] = s
    print(f"[{time.time()-T0:6.1f}s] grid {tag}: {s}")
    return out


# ------------------------------------------------------------------ Phase 0 checks
base = DemoB()
b0 = np.linspace(1, 6, 11); k0 = np.linspace(-3, 3, 11)
Bg, Kg = np.meshgrid(b0, k0, indexing="ij"); bg, kg = Bg.ravel(), Kg.ravel()
# (V1) invariance: start at S*, no label noise -> must stay exactly at S*
d0 = DemoB(sigma=0.0)
inv = []
for closed in (False, True):
    r = run(d0, bg, kg, closed, R=4, T=80, u0=(0.0, 0.0), keep_traj=True)
    inv.append(np.nanmax(np.abs(r["UB"])))
    inv.append(np.nanmax(np.abs(r["H"])))
NUM["invMax"] = float(max(inv))
# (V2) separability: cross-block derivative of the round map
def cross_jac(demo, eps=1e-6, reps=50):
    rng_seed = 3; vals = []
    for s in range(reps):
        w = np.array([[[1.4, 0.2], [1.4 + eps, 0.2]]]); z = np.zeros((1, 2))
        w2, _ = demo.step(w, z, np.random.default_rng(s), np.array([4.0]), np.array([1.0]), True)
        vals.append(abs(w2[0, 1, 1] - w2[0, 0, 1]) / eps)
    return float(np.max(vals))
NUM["crossSep"] = cross_jac(DemoB(separable=True))
NUM["crossNonSep"] = cross_jac(DemoB(separable=False))
# (V3) visibility: w_O^B = 0 by construction; (V4) guard g_V
NUM["gV"] = 1 - base.eta * base.cV
# (V5) linearity of rho estimate w.r.t. perturbation size
bl = np.array([1.5, 1.5, 5.0, 5.0]); kl = np.array([0.5, 2.5, -2.5, 1.0])
lin = {}
for eps in (1e-4, 1e-6, 1e-8):
    lin[eps] = [np.median(run(base, bl, kl, c, R=10, eps=eps, seed=5)["rho"], 1) for c in (False, True)]
NUM["linMaxLogDiff"] = float(max(np.max(np.abs(np.log(lin[e][i]) - np.log(lin[1e-6][i]))) for e in (1e-4, 1e-8) for i in (0, 1)))
RES["phase0"] = dict(inv=inv, lin=lin)
print(f"[{time.time()-T0:6.1f}s] phase0 {NUM}")

# ------------------------------------------------------------------ main grid
betas = np.linspace(1, 6, 21); kappas = np.linspace(-3, 3, 21)
RES["main"] = grid_experiment(base, betas, kappas, R=20, tag="main")

# ------------------------------------------------------------------ representative trajectories
reps = dict(II=(1.5, 0.5), ID=(1.5, 2.5), DI=(5.0, -2.5), DD=(5.0, 1.0))
traj = {}
for nm, (bb, kk) in reps.items():
    traj[nm] = {}
    for loop, closed in (("OL", False), ("CL", True)):
        r = run(base, np.array([bb]), np.array([kk]), closed, R=20, T=80, seed=11, keep_traj=True)
        traj[nm][loop] = dict(O=r["O"][0], H=r["H"][0], UB=r["UB"][0], Z=r["Z"][0], rho=r["rho"][0],
                              lab=aggregate(seed_labels(r))[0])
    rOL, rCL = base.predicted_rho(bb, kk)
    traj[nm]["pred"] = (float(rOL), float(rCL))
RES["traj"] = traj; RES["reps"] = reps
print(f"[{time.time()-T0:6.1f}s] trajectories done")

# ------------------------------------------------------------------ boundary sweeps
sweeps = {}
def sweep(name, bvals, kvals, closed, pred_cross):
    r = run(base, bvals, kvals, closed, R=20, T=80, seed=21)
    lab = aggregate(seed_labels(r)); rho = np.median(r["rho"], 1); q = np.quantile(r["rho"], [0.25, 0.75], axis=1)
    x = bvals if np.ptp(bvals) > 0 else kvals
    lr = np.log(rho); idx = np.where(np.sign(lr[:-1]) != np.sign(lr[1:]))[0]
    cross = float(x[idx[0]] - lr[idx[0]] * (x[idx[0] + 1] - x[idx[0]]) / (lr[idx[0] + 1] - lr[idx[0]])) if len(idx) else np.nan
    rOL, rCL = base.predicted_rho(bvals, kvals)
    sweeps[name] = dict(x=x, rho=rho, q=q, lab=lab, pred=(rCL if closed else rOL), cross=cross, pred_cross=pred_cross, closed=closed)
    print(f"[{time.time()-T0:6.1f}s] sweep {name}: crossing {cross:.3f} (pred {pred_cross:.3f})")
nsw = 41
sweep("OL_beta", np.linspace(2, 4, nsw), np.zeros(nsw), False, 3.0)
sweep("CL_kappa_beta2", np.full(nsw, 2.0), np.linspace(-0.5, 1.5, nsw), True, 1.5 - 0.5 * 2.0)
sweep("CL_kappa_beta5", np.full(nsw, 5.0), np.linspace(-2.0, 0.0, nsw), True, 1.5 - 0.5 * 5.0)
RES["sweeps"] = sweeps

# ------------------------------------------------------------------ ablations (13x13, R=10)
ab_b = np.linspace(1, 6, 13); ab_k = np.linspace(-3, 3, 13)
ablations = {
    "reference": DemoB(),
    "no_separability": DemoB(separable=False),
    "O_sees_B": DemoB(O_sees_B=True),
    "composition_feedback": DemoB(feedback="composition"),
    "frozen_OL_driver": DemoB(ol_driver="frozen"),
}
RES["ablations"] = {nm: grid_experiment(dm, ab_b, ab_k, R=10, seed=31, tag=nm) for nm, dm in ablations.items()}

# ------------------------------------------------------------------ robustness (13x13, R=10)
rob = {}
for n in (50, 200, 1000):
    rob[f"n={n}"] = grid_experiment(DemoB(n=n), ab_b, ab_k, R=10, seed=41, tag=f"n={n}")["summary"]
for T in (40, 160):
    rob[f"T={T}"] = grid_experiment(base, ab_b, ab_k, R=10, T=T, seed=41, tag=f"T={T}")["summary"]
for sg in (0.05, 0.3):
    rob[f"sigma={sg}"] = grid_experiment(DemoB(sigma=sg), ab_b, ab_k, R=10, seed=41, tag=f"sigma={sg}")["summary"]
RES["robust"] = rob

RES["num"] = NUM
pickle.dump(RES, open("results.pkl", "wb"))
print(f"[{time.time()-T0:6.1f}s] all done")
