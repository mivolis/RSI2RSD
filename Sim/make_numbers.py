import pickle, numpy as np
R = pickle.load(open("results.pkl", "rb"))
L = []
def mac(name, val): L.append(f"\\newcommand{{\\{name}}}{{{val}}}")
num = R["num"]
mac("invMax", f"{num['invMax']:.1f}")
mac("crossSep", f"{num['crossSep']:.1f}")
mac("crossNonSep", f"{num['crossNonSep']:.3f}")
mac("linMaxLogDiff", f"{num['linMaxLogDiff']:.0e}".replace("e-0", "e-").replace("e-", r"\times10^{-") + "}")
s = R["main"]["summary"]
for k in ("II", "ID", "DI", "DD"):
    mac(f"main{k}", s["counts"][k]); mac(f"pred{k}", s["pred_counts"][k])
mac("mainB", s["nB"]); mac("mainX", s["nX"]); mac("mainNear", s["n_near"]); mac("mainFar", s["n_far"])
far_agree_n = round(s["agree_far"] * s["n_far"])
mac("mainAgreeFar", f"{100*s['agree_far']:.1f}"); mac("mainAgreeFarN", far_agree_n)
mac("mainAgreeAll", f"{100*s['agree_all']:.1f}"); mac("mainCons", f"{100*s['cons']:.1f}")
mac("mainRhoErrMed", f"{100*s['rho_logerr_med']:.2f}"); mac("mainRhoErrPn", f"{100*s['rho_logerr_p95']:.2f}")
sw = R["sweeps"]
for nm, key in (("OL_beta", "SwA"), ("CL_kappa_beta2", "SwB"), ("CL_kappa_beta5", "SwC")):
    mac(f"cross{key}", f"{sw[nm]['cross']:.3f}"); mac(f"pcross{key}", f"{sw[nm]['pred_cross']:.3f}")
rows = []
for nm, lab in (("reference", "reference"), ("no_separability", "DG7 removed"), ("O_sees_B", "O3 removed"),
                ("composition_feedback", "DG6 removed"), ("frozen_OL_driver", "OL driver frozen")):
    g = R["ablations"][nm]["summary"]; c = g["counts"]
    rows.append(f"{lab} & {c['II']} & {c['ID']} & {c['DI']} & {c['DD']} & {g['nB']} & {g['nX']} & {100*g['agree_far']:.0f}\\% \\\\")
open("ablation_rows.tex", "w").write("\n".join(rows))
rows = []
for nm, g in R["robust"].items():
    c = g["counts"]
    lab = nm.replace("sigma", r"$\sigma$").replace("n=", "$n=").replace("T=", "$T=")
    if lab.startswith("$n=") or lab.startswith("$T="): lab = lab + "$"
    else: lab = lab.replace("=", "=")
    rows.append(f"{lab} & {c['II']} & {c['ID']} & {c['DI']} & {c['DD']} & {g['nB']} & {g['nX']} & {100*g['agree_far']:.1f}\\% & {100*g['rho_logerr_med']:.2f}\\% \\\\")
open("robust_rows.tex", "w").write("\n".join(rows))
# representative points
tr = R["traj"]
for reg in ("II", "ID", "DI", "DD"):
    for loop in ("OL", "CL"):
        mac(f"rho{reg}{loop}", f"{np.nanmedian(tr[reg][loop]['rho']):.2f}")
    mac(f"prho{reg}OL", f"{tr[reg]['pred'][0]:.2f}"); mac(f"prho{reg}CL", f"{tr[reg]['pred'][1]:.2f}")
open("numbers.tex", "w").write("\n".join(L) + "\n")
print(open("numbers.tex").read()); print(open("ablation_rows.tex").read()); print(open("robust_rows.tex").read())
