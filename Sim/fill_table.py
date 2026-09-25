"""Fill the application table: lambda in {0.3,0.5} x beta in {0.0005,0.001}, start after block 15.
Columns: share of updates with O_t>0 (own supervision), change in H with own supervision (CL),
change in H with replayed supervision (OL). Cells: mean +/- SD over 5 seeds; last row: mean [95% CI] over 20 runs.
Usage: python fill_table.py tao_seed_H.csv
tao_seed_H.csv columns: lambda,beta,seed,loop,H_R5   (lambda=0.3; loop CL = accept_closed, OL = accept_replay;
fork_b15_h1410; H_R5 = mean H over continuation blocks 11-15 minus blocks 1-5, H = -fixed-Q student CE)."""
import sys, numpy as np, pandas as pd
from scipy import stats
H={("CL",0.5,0.0005):[-0.383,-0.648,-0.273,-0.386,-0.237], ("CL",0.5,0.001):[-0.183,-0.117,-0.254,0.064,-0.025],
   ("OL",0.5,0.0005):[-0.341,0.045,-0.232,-0.177,-0.385], ("OL",0.5,0.001):[-0.094,-0.081,-0.461,-0.511,-0.108]}
KNOWN_MEAN={("CL",0.3,0.0005):-0.18,("CL",0.3,0.001):-0.20,("OL",0.3,0.0005):-0.26,("OL",0.3,0.001):-0.03}
POOLED_MEAN={"CL":-0.22,"OL":-0.19}
O={(0.3,0.0005):(99.83,0.10),(0.3,0.001):(99.77,0.13),(0.5,0.0005):(99.74,0.14),(0.5,0.001):(99.72,0.07)}
O_all=99.77
if len(sys.argv)>1:
    t=pd.read_csv(sys.argv[1])
    for lp in ("CL","OL"):
        for b in (0.0005,0.001):
            H[(lp,0.3,b)]=t[(t.loop==lp)&np.isclose(t["lambda"],0.3)&np.isclose(t.beta,b)].H_R5.tolist()
def cell(lp,l,b):
    v=H.get((lp,l,b))
    if v: v=np.array(v); return f"${v.mean():.2f} \\pm {v.std(ddof=1):.2f}$"
    return f"${KNOWN_MEAN[(lp,l,b)]:.2f} \\pm$ \\textbf{{[SD]}}"
def pooled(lp):
    ks=[(lp,l,b) for l in (0.3,0.5) for b in (0.0005,0.001)]
    if all(k in H for k in ks):
        v=np.concatenate([H[k] for k in ks]); m=v.mean(); hw=stats.t.ppf(.975,len(v)-1)*v.std(ddof=1)/np.sqrt(len(v))
        return "$%.2f$ $[%.2f,\\,%.2f]$" % (m, m-hw, m+hw)
    return f"${POOLED_MEAN[lp]:.2f}$ \\textbf{{[CI]}}"
rows=[f"{l:g} & {b:g} & ${O[(l,b)][0]:.2f} \\pm {O[(l,b)][1]:.2f}$ & {cell('CL',l,b)} & {cell('OL',l,b)} \\\\" for l in (0.3,0.5) for b in (0.0005,0.001)]
rows+=["\\midrule",f"\\multicolumn{{2}}{{l}}{{All 20 runs}} & ${O_all:.2f}$ & {pooled('CL')} & {pooled('OL')} \\\\"]
print("\n".join(rows))
