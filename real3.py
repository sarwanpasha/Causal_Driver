import csv,json,os,re,collections
import numpy as np,torch
from real_ref import zscore,load_edges,edge_cons,scaled
import oriented as O
L=1e-2;B_BOOT=2000
def C0(B):
    A=B.abs();d=A.diagonal().clone();q=d[d>0]
    md=q.median() if q.numel() else torch.tensor(1.,device=B.device,dtype=B.dtype)
    d=torch.where(d>1e-12*md,d,md);C=A/d[None,:];C.fill_diagonal_(0.);return C
def fas(B):
    P=C0(B).T.cpu().numpy();p=P.shape[0];l=[];r=[];rem=list(range(p))
    while rem:
        i=np.array(rem);s=P[np.ix_(i,i)];o=s.sum(1);n=s.sum(0)
        k=np.where(o<=0)[0]
        if k.size:v=int(i[k[0]]);r.append(v);rem.remove(v);continue
        k=np.where(n<=0)[0]
        if k.size:v=int(i[k[0]]);l.append(v);rem.remove(v);continue
        v=int(i[int(np.argmax(o-n))]);l.append(v);rem.remove(v)
    return np.array(l+r[::-1])
def fit(Xz,D,dev,lam=L):
    return O.fit_B(torch.from_numpy(Xz.astype(np.float32)).to(dev),
                   torch.from_numpy(np.nan_to_num(D).astype(np.float32)).to(dev),lam=lam)
def ec_of(order,E):
    rank=np.empty(len(order),dtype=np.int64);rank[np.asarray(order)]=np.arange(len(order))
    return float(np.mean([rank[a]<rank[b] for a,b in E]))
def fam(s):
    m=re.match(r'^([A-Z]{2,}?)[0-9]*[A-Z]?$',s)
    return m.group(1) if m and len(m.group(1))>=3 else None
if __name__=="__main__":
    g=np.random.default_rng(0)
    z=np.load("cache/panel2000_depvar.npz",allow_pickle=True)
    sym=list(map(str,z["symbols"]));Xr=np.asarray(z["X_s"],dtype=np.float64)
    Ds=np.asarray(z["D_s"],dtype=np.float64)
    E=load_edges("ref/trrust_rawdata.human.tsv",sym);Xz=zscore(Xr)
    dev="cuda" if torch.cuda.is_available() else "cpu";rows=[]
    def rec(**k): rows.append(k);print(json.dumps(k),flush=True)
    D=scaled(Ds,Xr,1.0,"rescaled",g)
    B=fit(Xz,D,dev)
    o_fas=fas(B)
    colmean=np.argsort(-np.nan_to_num(D).mean(0))
    base_fas=ec_of(o_fas,E);base_cm=ec_of(colmean,E)
    rec(test="point",rule="fas",ec=base_fas);rec(test="point",rule="colmean",ec=base_cm)
    # regulator-blocked bootstrap and paired difference
    byreg=collections.defaultdict(list)
    for a,b in E: byreg[a].append((a,b))
    regs=list(byreg)
    rf=np.empty(B_BOOT);rc=np.empty(B_BOOT)
    rank_f=np.empty(len(o_fas),dtype=np.int64);rank_f[o_fas]=np.arange(len(o_fas))
    rank_c=np.empty(len(colmean),dtype=np.int64);rank_c[colmean]=np.arange(len(colmean))
    for t in range(B_BOOT):
        pick=g.choice(len(regs),len(regs),replace=True)
        ed=[e for i in pick for e in byreg[regs[i]]]
        rf[t]=np.mean([rank_f[a]<rank_f[b] for a,b in ed])
        rc[t]=np.mean([rank_c[a]<rank_c[b] for a,b in ed])
    d=rf-rc
    rec(test="regulator_bootstrap",n_regulators=len(regs),reps=B_BOOT,
        fas_lo=float(np.percentile(rf,2.5)),fas_hi=float(np.percentile(rf,97.5)),
        cm_lo=float(np.percentile(rc,2.5)),cm_hi=float(np.percentile(rc,97.5)),
        diff=float(d.mean()),diff_lo=float(np.percentile(d,2.5)),
        diff_hi=float(np.percentile(d,97.5)),p_two_sided=float(2*min((d<=0).mean(),(d>=0).mean())))
    # paralog proxy: blank putative same-family pairs in Bhat before ordering
    fams=collections.defaultdict(list)
    for i,s in enumerate(sym):
        f=fam(s)
        if f: fams[f].append(i)
    pairs=[(a,b) for v in fams.values() if 2<=len(v)<=12 for a in v for b in v if a!=b]
    Bm=B.clone()
    if pairs:
        ii=torch.tensor([a for a,_ in pairs]);jj=torch.tensor([b for _,b in pairs])
        Bm[ii,jj]=0.0
    rec(test="paralog_blanked",n_family_pairs=len(pairs),
        n_families=sum(1 for v in fams.values() if 2<=len(v)<=12),
        ec=ec_of(fas(Bm),E))
    # simple baselines
    dv=np.nanstd(Ds,axis=0);ex=np.nanmean(Xr,axis=0)
    Ab=B.abs().cpu().numpy();np.fill_diagonal(Ab,0.0)
    rec(test="baseline",name="pairwise_unnormalized",ec=ec_of(np.argsort(-np.sign(Ab-Ab.T).sum(1)),E))
    rec(test="baseline",name="dependency_variance",ec=ec_of(np.argsort(-dv),E))
    rec(test="baseline",name="expression_mean",ec=ec_of(np.argsort(-ex),E))
    rec(test="baseline",name="covariate_variance",ec=ec_of(np.argsort(np.nanstd(Xr,axis=0)),E))
    # ridge sensitivity
    for lam in (1e-3,1e-2,1e-1,1.0):
        rec(test="lambda",lam=lam,ec=ec_of(fas(fit(Xz,D,dev,lam)),E))
    os.makedirs("real3_out",exist_ok=True)
    k=sorted({x for r in rows for x in r})
    with open("real3_out/real3.csv","w",newline="") as fh:
        w=csv.DictWriter(fh,fieldnames=k);w.writeheader();w.writerows(rows)
    print("DONE_REAL3")
