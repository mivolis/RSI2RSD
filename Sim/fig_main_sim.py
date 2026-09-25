"""One composite figure for the simulation section (Q1-Q3). Sharpening is written s (was beta in the report)."""
import numpy as np, pickle, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt, matplotlib.ticker as mt
from matplotlib.patches import Rectangle, Patch
from matplotlib.lines import Line2D
R=pickle.load(open("results.pkl","rb")); m=R["main"]; tr=R["traj"]; reps=R["reps"]
INK,INK2,MUTED,GRID,SURF="#0b0b0b","#52514e","#8a8983","#e4e3df","#ffffff"
COL={"II":"#2a78d6","ID":"#eb6834","DI":"#1baf7a","DD":"#4a3aa7"}; TXT={"II":"white","ID":"white","DI":INK,"DD":"white"}
B_COL,X_COL="#e4e3df","#b4b2a9"; OLC="#9a9994"; CLC=INK
plt.rcParams.update({"font.size":6.8,"font.family":"DejaVu Sans","axes.edgecolor":MUTED,"axes.labelcolor":INK2,
  "xtick.color":INK2,"ytick.color":INK2,"axes.titlesize":7.2,"axes.titlecolor":INK,"axes.spines.top":False,
  "axes.spines.right":False,"pdf.fonttype":42,"xtick.major.size":2.5,"ytick.major.size":2.5})
fig=plt.figure(figsize=(5.5,2.95))
gs=fig.add_gridspec(2,3,width_ratios=[1.25,1,1.0],wspace=0.5,hspace=0.7,left=0.085,right=0.985,top=0.92,bottom=0.31)
# (a) region map ---------------------------------------------------------------
ax=fig.add_subplot(gs[:,0])
b,k=0.4+0.2*m["betas"],0.4*m["kappas"]; db,dk=b[1]-b[0],k[1]-k[0]
for i,(x,y) in enumerate(zip(0.4+0.2*m["b"],0.4*m["k"])):
    reg=m["region"][i]; lo,lc=m["OL"]["lab"][i],m["CL"]["lab"][i]
    if reg in COL: fc,h=COL[reg],None
    elif "X" in (lo,lc): fc,h=X_COL,"////"
    else: fc,h=B_COL,None
    ax.add_patch(Rectangle((x-db/2,y-dk/2),db,dk,facecolor=fc,edgecolor=SURF,lw=0.5,hatch=h))
ax.axvline(1.0,color=INK,ls="--",lw=1.0); bb=np.linspace(b[0]-db/2,b[-1]+db/2,50); ax.plot(bb,1.0-bb,color=INK,lw=1.0)
ax.set_xlim(b[0]-db/2,b[-1]+db/2); ax.set_ylim(k[0]-dk/2,k[-1]+dk/2)
for reg,(x,y) in {"II":(0.78,-0.76),"ID":(0.87,0.62),"DI":(1.19,-0.92),"DD":(1.36,0.68)}.items():
    ax.text(x,y,r"$\mathcal{R}_{\mathrm{%s}}$"%reg,ha="center",va="center",fontsize=10,color=TXT[reg],weight="bold")
for reg,lab in (("ID","b"),("DI","c")):
    x,y=reps[reg]; x,y=0.4+0.2*x,0.4*y; ax.plot(x,y,marker="o",ms=9,mfc="white",mec=INK,mew=1.0,zorder=5); ax.text(x,y,lab,ha="center",va="center",fontsize=7,weight="bold",zorder=6)
ax.set_xlabel(r"internal gain $a$ (sharpening $s$)"); ax.set_ylabel(r"environmental return $g$ (feedback $\kappa$)")
ax.set_title(r"(a) regions vs. predicted boundaries",loc="left")
# (b),(c) trajectories -----------------------------------------------------------
def traj(ax,reg,title):
    for loop,c in (("OL",OLC),("CL",CLC)):
        d=tr[reg][loop]; y=-d["H"]; t=np.arange(y.shape[1]); med=np.nanmedian(y,0); lo,hi=np.nanquantile(y,[0.1,0.9],axis=0)
        ax.fill_between(t,lo,hi,color=c,alpha=0.15,lw=0); ax.plot(t,med,color=c,lw=1.4)
    y=-tr[reg]["CL"]["O"]; t=np.arange(y.shape[1]); ax.plot(t,np.nanmedian(y,0),color=MUTED,lw=1.0,ls=(0,(2,1.5)))
    ax.set_yscale("log"); ax.set_ylim(1e-6,1e12); ax.set_xlim(0,80)
    ax.yaxis.set_major_locator(mt.LogLocator(numticks=4)); ax.yaxis.set_minor_locator(mt.NullLocator())
    ax.grid(True,axis="y",color=GRID,lw=0.5); ax.set_axisbelow(True)
    ax.set_title(title,loc="left"); ax.set_ylabel("excess loss")
axb=fig.add_subplot(gs[0,1]); traj(axb,"ID",r"(b) reinforcing ($\mathcal{R}_{\mathrm{ID}}$)")
axc=fig.add_subplot(gs[1,1]); traj(axc,"DI",r"(c) compensating ($\mathcal{R}_{\mathrm{DI}}$)"); axc.set_xlabel("round $t$")
sb,kb=reps["ID"]; sc,kc=reps["DI"]
axb.text(79,1e5,f"$a$={0.4+0.2*sb:.1f}, $g$={0.4*kb:.1f}",ha="right",fontsize=6,color=INK2)
axc.text(79,1e5,f"$a$={0.4+0.2*sc:.1f}, $g$={0.4*kc:.1f}",ha="right",fontsize=6,color=INK2)
# (d) measured vs predicted rho --------------------------------------------------
ax=fig.add_subplot(gs[:,2])
for loop,mk,fc,ec,s,z in (("CL","s",INK2,INK2,9,3),("OL","o","none",INK,20,4)):
    ax.scatter(m[f"r{loop}_pred"],m[loop]["rho"],s=s,marker=mk,facecolor=fc,edgecolor=ec,lw=0.7,zorder=z)
lim=[0.5,2.4]; ax.plot(lim,lim,color=MUTED,lw=1.0,zorder=2); ax.axhline(1,color=INK,lw=0.6,ls=":"); ax.axvline(1,color=INK,lw=0.6,ls=":")
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(lim); ax.set_ylim(lim)
for a in (ax.xaxis,ax.yaxis): a.set_major_formatter(mt.FormatStrFormatter("%.1f")); a.set_minor_formatter(mt.NullFormatter())
ax.set_xticks([0.6,1.0,1.5,2.0]); ax.set_yticks([0.6,1.0,1.5,2.0])
ax.set_xlabel(r"predicted spectral radius $\varrho$"); ax.set_ylabel(r"measured amplification $\hat\varrho$")
ax.grid(True,color=GRID,lw=0.5); ax.set_axisbelow(True)
s=m["summary"]; ax.text(0.53,2.12,f"median error {100*s['rho_logerr_med']:.2f}%",fontsize=6,color=INK2)
ax.text(1.05,0.53,"contracting | amplifying",fontsize=6,color=MUTED,ha="center")
ax.set_title(r"(d) $\hat\varrho$ vs. predicted $\varrho$",loc="left")
# legend -------------------------------------------------------------------------
h=[Patch(color=COL[r],label=r"$\mathcal{R}_{\mathrm{%s}}$"%r) for r in COL]+[Patch(facecolor=B_COL,label="undetermined"),Patch(facecolor=X_COL,hatch="////",label="inconsistent")]
h+=[Line2D([],[],color=INK,ls="--",lw=1,label=r"$a=1$"),Line2D([],[],color=INK,lw=1,label=r"$a+g=1$"),
    Line2D([],[],color=OLC,lw=1.4,label=r"OL: protected loss"),Line2D([],[],color=CLC,lw=1.4,label=r"CL: protected loss"),
    Line2D([],[],color=MUTED,lw=1,ls=(0,(2,1.5)),label=r"evaluated loss (both loops)"),
    Line2D([],[],marker="o",ls="",mfc="none",mec=INK,ms=4,label=r"OL setting"),Line2D([],[],marker="s",ls="",mfc=INK2,mec=INK2,ms=3.5,label=r"CL setting")]
fig.legend(handles=h,loc="lower center",ncol=5,frameon=False,fontsize=6,bbox_to_anchor=(0.52,-0.01),columnspacing=1.0,handlelength=1.8)
fig.savefig("fig_sim_main.pdf"); fig.savefig("fig_sim_main.png",dpi=220); print("ok")
