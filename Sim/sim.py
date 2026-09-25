
"""
Simulation

Learner      w = (w_V, w_B), linear predictor f(x) = w . x, x in R^2.
Truth        w* = (1, 0): coordinate V is informative, coordinate B is spurious (true weight 0).
Round t:     pool of four components (each n samples, x ~ N(0, I) restricted by an input mask)
               V-true : y = w*.x + sigma*eps                       weight c_V
               B-true : y = w*.x + sigma*eps                       weight c_T
               B-self : y = w_{V,t} x_V + beta * w_{B,t} x_B        weight c_S   (sharpened self-targets)
               B-env  : y = w*_V x_V + zeta_t x_B + sigma*eps       weight c_E   (environment labels)
             update  : one gradient step (K = 1, step eta) on sum_c c_c * mean (1/2)(w.x - y)^2
             inherit : w_{t+1} is passed on unchanged
             env     : zeta_{t+1} = omega*zeta_t + kappa*q_{t+1},  q = w_{B,t+1} (CL) or 0 (OL)
Metrics      O = -(excess risk on V-inputs) = -u_V^2 ;  H = -(excess risk on V- and B-inputs) = -(u_V^2 + u_B^2)
"""
import numpy as np


class DemoB:
    def __init__(self, eta=0.5, cV=0.8, cT=0.4, cS=0.4, cE=0.4, omega=0.5, sigma=0.1, n=200,
                 u0=(1.0, 0.3), cap=1e6, separable=True, O_sees_B=False, feedback="label", ol_driver="target"):
        self.eta, self.cV, self.cT, self.cS, self.cE = eta, cV, cT, cS, cE
        self.om, self.sig, self.n = omega, sigma, n
        self.w_star = np.array([1.0, 0.0]); self.u0 = np.array(u0, float); self.cap = cap
        self.separable, self.O_sees_B, self.feedback, self.ol_driver = separable, O_sees_B, feedback, ol_driver

    # ---------------------------------------------------------------- reduced coordinates (expected linearization)
    def reduced(self, beta, kappa):
        beta = np.asarray(beta, float); kappa = np.asarray(kappa, float)
        a = 1 - self.eta * (self.cT + self.cE) + self.eta * self.cS * (beta - 1)
        chi = self.eta * self.cE * np.ones_like(a)
        gV = 1 - self.eta * self.cV
        return a, chi, kappa * np.ones_like(a), self.om, gV

    def predicted_rho(self, beta, kappa):
        a, chi, kap, om, gV = self.reduced(beta, kappa)
        a, chi, kap = np.broadcast_arrays(a, chi, kap)
        rOL = np.maximum.reduce([np.abs(a), np.full_like(a, abs(gV)), np.full_like(a, om)])
        tr = a + om + kap * chi; det = a * om
        disc = tr**2 - 4 * det
        sq = np.sqrt(np.abs(disc))
        rCL = np.where(disc >= 0, np.maximum(np.abs(tr + sq), np.abs(tr - sq)) / 2, np.sqrt(np.abs(det)))
        rCL = np.maximum(rCL, abs(gV))
        return rOL, rCL

    # ---------------------------------------------------------------- one round, vectorized over batch and twins
    def _X(self, rng, Bn, which):
        X = rng.normal(size=(Bn, self.n, 2))
        if self.separable:
            X[..., 1 if which == "V" else 0] = 0.0
        return X

    def step(self, w, zeta, rng, beta, kappa, closed):
        """w: (Bn, 2 twins, 2), zeta: (Bn, 2). Random draws are shared across twins (CRN)."""
        Bn, n, eta = w.shape[0], self.n, self.eta
        XV, XT, XS, XE = (self._X(rng, Bn, k) for k in ("V", "B", "B", "B"))
        eV, eT, eE = (rng.normal(size=(Bn, 1, n)) for _ in range(3))
        ws = self.w_star
        pred = lambda X: np.einsum("bnd,btd->btn", X, w)
        grad = lambda X, r: np.einsum("bnd,btn->btd", X, r) / n
        yV = np.einsum("bnd,d->bn", XV, ws)[:, None] + self.sig * eV
        yT = np.einsum("bnd,d->bn", XT, ws)[:, None] + self.sig * eT
        wsharp = np.stack([w[..., 0], beta[:, None] * w[..., 1]], -1)            # sharpened self-targets
        yS = np.einsum("bnd,btd->btn", XS, wsharp)
        if self.feedback == "label":
            yE = XE[..., 0][:, None] * ws[0] + zeta[..., None] * XE[..., 1][:, None] + self.sig * eE
            cE = self.cE
        else:  # composition-only: truthful labels, driver changes how much B data is included
            yE = np.einsum("bnd,d->bn", XE, ws)[:, None] + self.sig * eE
            cE = self.cE * (1 + np.tanh(zeta))[..., None]
        g = (self.cV * grad(XV, pred(XV) - yV) + self.cT * grad(XT, pred(XT) - yT)
             + self.cS * grad(XS, pred(XS) - yS) + cE * grad(XE, pred(XE) - yE))
        w2 = w - eta * g
        if closed:
            q = w2[..., 1]
        elif self.ol_driver == "frozen":
            q = self.w_star[1] + self.u0[1]
        else:
            q = self.w_star[1]
        z2 = self.om * zeta + kappa[:, None] * q
        return w2, z2

    def metrics(self, w):
        u = w - self.w_star
        H = -(u[..., 0]**2 + u[..., 1]**2)
        O = H.copy() if self.O_sees_B else -(u[..., 0]**2)
        return O, H


# -------------------------------------------------------------------- simulator
def run(demo, beta, kappa, closed, R=20, T=80, t1=15, Wmin=10, win=5, eps=1e-6, seed=0, u0=None, keep_traj=False):
    G = len(beta); Bn = G * R
    b, k = np.repeat(beta, R), np.repeat(kappa, R)
    rng = np.random.default_rng(seed)
    u0 = demo.u0 if u0 is None else np.asarray(u0, float)
    w = np.tile(demo.w_star + u0, (Bn, 1)); w = np.stack([w, w], 1); zeta = np.zeros((Bn, 2))
    d = rng.normal(size=(Bn, 3)); d /= np.linalg.norm(d, axis=1, keepdims=True)
    scale = np.maximum(1.0, np.linalg.norm(np.c_[w[:, 0], zeta[:, 0]], axis=1))
    w[:, 1] += eps * scale[:, None] * d[:, :2]; zeta[:, 1] += eps * scale * d[:, 2]
    O = np.full((Bn, T + 1), np.nan); H = np.full((Bn, T + 1), np.nan)
    UB = np.full((Bn, T + 1), np.nan); Z = np.full((Bn, T + 1), np.nan)
    O[:, 0], H[:, 0] = demo.metrics(w[:, 0]); UB[:, 0] = w[:, 0, 1] - demo.w_star[1]; Z[:, 0] = 0
    LG = np.full((Bn, T), np.nan); alive = np.ones(Bn, bool); tau = np.full(Bn, T)
    for t in range(T):
        w2, z2 = demo.step(w, zeta, rng, b, k, closed)
        ref = np.c_[w2[:, 0], z2[:, 0]]; diff = np.c_[w2[:, 1], z2[:, 1]] - ref
        nd = np.linalg.norm(diff, axis=1); g = nd / scale
        ok = alive & (g > 0) & np.isfinite(g); LG[ok, t] = np.log(g[ok] / eps)
        scale = np.maximum(1.0, np.linalg.norm(ref, axis=1))
        dn = diff / np.maximum(nd, 1e-300)[:, None]
        w2[:, 1] = w2[:, 0] + eps * scale[:, None] * dn[:, :2]; z2[:, 1] = z2[:, 0] + eps * scale * dn[:, 2]
        w = np.where(alive[:, None, None], w2, w); zeta = np.where(alive[:, None], z2, zeta)
        o, h = demo.metrics(w[:, 0])
        O[alive, t + 1], H[alive, t + 1] = o[alive], h[alive]
        UB[alive, t + 1] = w[alive, 0, 1] - demo.w_star[1]; Z[alive, t + 1] = zeta[alive, 0]
        hit = alive & (np.abs(w[:, 0, 1] - demo.w_star[1]) >= demo.cap); tau[hit] = t + 1; alive &= ~hit
    rho = np.array([np.exp(np.nanmean(LG[i, min(t1, max(tau[i] - Wmin, 0)):tau[i]])) for i in range(Bn)])
    TO = np.array([np.nanmean(O[i, max(tau[i] - win + 1, 0):tau[i] + 1]) - np.nanmean(O[i, :win]) for i in range(Bn)])
    TH = np.array([np.nanmean(H[i, max(tau[i] - win + 1, 0):tau[i] + 1]) - np.nanmean(H[i, :win]) for i in range(Bn)])
    rs = lambda a: a.reshape(G, R, *a.shape[1:])
    out = dict(TO=rs(TO), TH=rs(TH), rho=rs(rho), tau=rs(tau))
    if keep_traj:
        out.update(O=rs(O), H=rs(H), UB=rs(UB), Z=rs(Z))
    return out


def seed_labels(res):
    TO, TH, r = res["TO"], res["TH"], res["rho"]
    lab = np.full(TO.shape, "X", dtype=object)
    lab[(TO > 0) & (TH > 0) & (r < 1)] = "I"
    lab[(TO > 0) & (TH < 0) & (r > 1)] = "D"
    return lab


def aggregate(lab, q=0.75):
    out = []
    for row in lab:
        vals, cnts = np.unique(row, return_counts=True); j = np.argmax(cnts); share = cnts[j] / len(row)
        out.append(vals[j] if share >= q else "B")
    return np.array(out, dtype=object)
