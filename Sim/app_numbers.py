"""All numbers for the application section and its appendix."""
import pandas as pd, numpy as np, json
from scipy import stats
U="/root/.claude/uploads/d37690a1-725e-5592-b293-84b80620b25d/"
tao=pd.read_csv(U+"b26b8d01-tao_pair01_pair02_seed_metrics.csv")
y05=pd.read_csv(U+"6111a4de-lambda05_seed_metrics.csv")
old=pd.read_csv(U+"06d220ef-all_settings_O_H_table.csv")
traj=pd.read_csv(U+"136a5886-O_positivity_by_trajectory.csv")
out={}
def ci(x):
    x=np.asarray(x,float); m=x.mean(); hw=stats.t.ppf(.975,len(x)-1)*x.std(ddof=1)/np.sqrt(len(x)); return m,m-hw,m+hw
cols=["lambda_value","beta","seed","fork","branch","loop","H_region5","H_terminal","H_window3","H_theil_sen","H_full15","H_local_rise_then_decline","H_early_gain_then_fall"]
D=pd.concat([tao[cols],y05[cols]])
main=D[(D.fork=="fork_b15_h1410")&D.beta.isin([0.0005,0.001])&D.branch.isin(["accept_closed","accept_replay"])]
# --- B.2 trend statistics, main 2x2 design
rows=[]
for lp in ["CL","OL"]:
    x=main[main.loop==lp]
    for st,lab in [("H_region5","R5"),("H_terminal","T"),("H_window3","W3"),("H_theil_sen","TS"),("H_full15","E15")]:
        v=x[st].values; m,lo,hi=ci(v)
        rows.append(dict(loop=lp,stat=lab,mean=m,lo=lo,hi=hi,neg=int((v<0).sum()),n=len(v)))
T2=pd.DataFrame(rows); print(T2.round(3).to_string()); T2.to_csv("B2_trend_statistics.csv",index=False)
# --- B.3 checkpoint after block 5 (closed loop), lambda=0.5 per seed; tao Pair03 (beta=0.0015) setting-level
b5=y05[(y05.fork=="fork_b5_h1410")&(y05.branch=="accept_closed")&y05.beta.isin([0.0005,0.001])]
r=[]
for b,g in b5.groupby("beta"):
    m,lo,hi=ci(g.H_region5); r.append(dict(lam=0.5,beta=b,mean=m,sd=g.H_region5.std(ddof=1),neg=int((g.H_region5<0).sum()),n=len(g),gtf=int(g.H_early_gain_then_fall.sum())))
p3=old[(old.owner=="tao")&(old.pair=="Pair03")&(old.loop=="CL")].iloc[0]
r.append(dict(lam=0.3,beta=0.0015,mean=p3.H_mean_late_minus_early,sd=np.nan,neg=5,n=5,gtf=4))
T3=pd.DataFrame(r); print(T3.round(3)); T3.to_csv("B3_checkpoint5.csv",index=False)
allb5=b5.H_region5.values; print("b5 lambda0.5 pooled",np.round(ci(allb5),3),(allb5<0).sum(),len(allb5))
# --- B.4 O positivity, all long-fork trajectories
out["O_all_pos"]=int(traj.O_positive.sum()); out["O_all_tot"]=int(traj.O_updates.sum()); out["O_all_traj"]=len(traj)
out["O_all_rate"]=traj.O_positive.sum()/traj.O_updates.sum(); out["O_all_minfrac"]=traj.positive_fraction.min()
out["O_allpos_traj"]=int(traj.all_updates_positive.sum()); out["O_max_neg"]=int(traj.O_negative.max()); out["O_med_neg"]=float(traj.O_negative.median())
out["O_block_R5up_b5"]=int(traj[traj.fork=="fork_b5_h1410"].O_R5_up.sum()); out["O_block_R5up_b5_n"]=int((traj.fork=="fork_b5_h1410").sum())
out["O_block_R5up_b15"]=int(traj[traj.fork=="fork_b15_h1410"].O_R5_up.sum()); out["O_block_R5up_b15_n"]=int((traj.fork=="fork_b15_h1410").sum())
# --- B.5 other mixture weights, H only (Yingchuan matched rows)
yr=[]
for pair in ["Pair03","Pair04","Pair05"]:
    g=old[(old.owner=="yingchuan")&(old.pair==pair)]
    for lp in ["CL","OL"]:
        q=g[g.loop==lp].iloc[0]
        import re
        mm=re.search(r"R5 V(\d)/(\d)",q.H_top3_short)
        yr.append(dict(lam=q.lambda_value,beta=q.beta,fork=q.fork,loop=lp,R5=q.H_mean_late_minus_early,neg=int(mm.group(1)) if mm else None,n=5))
T5=pd.DataFrame(yr); print(T5); T5.to_csv("B5_other_lambda.csv",index=False)
# --- B.6 20 matched pairs
P=main.pivot_table(index=["lambda_value","beta","seed"],columns="loop",values=["H_region5","H_terminal"])
Q=pd.DataFrame({"OL_R5":P["H_region5"]["OL"],"CL_R5":P["H_region5"]["CL"],"OL_T":P["H_terminal"]["OL"],"CL_T":P["H_terminal"]["CL"]})
Q["gap_R5"]=Q.CL_R5-Q.OL_R5; Q["gap_T"]=Q.CL_T-Q.OL_T
print(Q.round(3).to_string()); Q.to_csv("B6_matched_pairs.csv")
out["gap_R5_CLworse"]=int((Q.gap_R5<0).sum()); out["gap_T_CLworse"]=int((Q.gap_T<0).sum())
out["gap_med_abs"]=float(Q.gap_R5.abs().median())
out["sign_agree_R5_T"]=int((np.sign(Q.gap_R5)==np.sign(Q.gap_T)).sum())
json.dump({k:(float(v) if not isinstance(v,int) else v) for k,v in out.items()},open("app_extra_numbers.json","w"),indent=1)
print(json.dumps(out,indent=1,default=float))
