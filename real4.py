import json,collections,numpy as np
from multiprocessing import Pool
exec(open('real3.py').read().split('if __name__')[0])
B_BOOT=2000; NSUB=300
rows=[]
def rec(**k): rows.append(k); print(json.dumps(k),flush=True)

def load_collectri(path,sym,mode='all'):
    idx={s:i for i,s in enumerate(sym)}; E=[]
    with open(path) as f:
        h=f.readline().rstrip('\n').split('\t')
        ks=h.index('source_genesymbol'); kt=h.index('target_genesymbol')
        kd=h.index('is_directed'); kr=h.index('sources')
        for ln in f:
            p=ln.rstrip('\n').split('\t')
            if len(p)<=max(ks,kt,kd,kr): continue
            if p[kd] not in ('True','1','true'): continue
            if mode=='notrrust' and 'TRRUST' in p[kr]: continue
            a=idx.get(p[ks]); b=idx.get(p[kt])
            if a is None or b is None or a==b: continue
            E.append((a,b))
    return sorted(set(E))

def blocks(E):
    d=collections.defaultdict(list)
    for a,b in E: d[a].append((a,b))
    return d

def correct(order,E):
    r=np.empty(len(order),dtype=np.int64); r[order]=np.arange(len(order))
    return np.array([1.0 if r[a]<r[b] else 0.0 for a,b in E])

def icc_of(dvals,bid):
    K=bid.max()+1; n=len(dvals); gm=dvals.mean()
    cnt=np.bincount(bid,minlength=K).astype(float)
    bm=np.bincount(bid,weights=dvals,minlength=K)/np.maximum(cnt,1)
    msb=(cnt*(bm-gm)**2).sum()/max(K-1,1)
    msw=((dvals-bm[bid])**2).sum()/max(n-K,1)
    m0=(n-(cnt**2).sum()/n)/max(K-1,1)
    return float((msb-msw)/max(msb+(m0-1)*msw,1e-12))

def boot_ci(d,bid,reps,seed):
    rng=np.random.default_rng(seed); K=bid.max()+1
    s=np.bincount(bid,weights=d,minlength=K); c=np.bincount(bid,minlength=K).astype(float)
    ix=rng.integers(0,K,(reps,K))
    v=s[ix].sum(1)/np.maximum(c[ix].sum(1),1e-12)
    lo,hi=np.percentile(v,[2.5,97.5])
    p=float(2*min((v<=0).mean(),(v>=0).mean()))
    return float(d.mean()),float(lo),float(hi),float(v.std()),p

def _sub(args):
    K,seed,dv,bid=args
    rng=np.random.default_rng(seed); regs=np.unique(bid)
    pick=rng.choice(regs,size=min(K,len(regs)),replace=(K>len(regs)))
    keep=np.concatenate([np.where(bid==r)[0] for r in pick])
    nb=np.concatenate([np.full((bid==r).sum(),i) for i,r in enumerate(pick)])
    _,lo,hi,_,_=boot_ci(dv[keep],nb,500,seed+7)
    return int(lo>0 or hi<0)

if __name__=="__main__":
    g=np.random.default_rng(0)
    z=np.load("cache/panel2000_depvar.npz",allow_pickle=True)
    sym=list(map(str,z["symbols"])); Xr=np.asarray(z["X_s"],dtype=np.float64)
    Ds=np.asarray(z["D_s"],dtype=np.float64); Xz=zscore(Xr)
    dev="cuda" if torch.cuda.is_available() else "cpu"
    D=scaled(Ds,Xr,1.0,"rescaled",g); B=fit(Xz,D,dev)
    o_fas=fas(B); colmean=np.argsort(-np.nan_to_num(D).mean(0))
    REFS={'trrust':load_edges("ref/trrust_rawdata.human.tsv",sym),
          'collectri':load_collectri("ref/collectri.tsv",sym,'all'),
          'collectri_no_trrust':load_collectri("ref/collectri.tsv",sym,'notrrust')}
    for name,E in REFS.items():
        if len(E)<20: rec(ref=name,error="too few edges",n=len(E)); continue
        bid=np.array([a for a,_ in E]); u={r:i for i,r in enumerate(sorted(set(bid.tolist())))}
        bid=np.array([u[a] for a in bid])
        cf=correct(o_fas,E); cc=correct(colmean,E); dv=cf-cc
        m,lo,hi,sd,pv=boot_ci(dv,bid,B_BOOT,11)
        rec(test="ref",ref=name,n_edges=len(E),n_regulators=int(bid.max()+1),
            ec_fas=float(cf.mean()),ec_colmean=float(cc.mean()),diff=m,diff_lo=lo,
            diff_hi=hi,boot_sd=sd,p_two_sided=pv,icc=icc_of(dv,bid),
            edges_per_regulator=float(len(E)/(bid.max()+1)))
        jobs=[]
        for K in [25,50,105,200,400,800,1600,3200]:
            for s in range(NSUB): jobs.append((K,90000+1000*K+s,dv,bid))
        with Pool(8) as pool: out=pool.map(_sub,jobs,chunksize=10)
        i=0
        for K in [25,50,105,200,400,800,1600,3200]:
            rec(test="power_curve",ref=name,regulators=K,
                detection_rate=float(np.mean(out[i:i+NSUB]))); i+=NSUB
    # permutation control on the two new references, pipeline refit from scratch
    n=Xz.shape[0]
    for name in ['collectri','collectri_no_trrust']:
        E=REFS[name]
        if len(E)<20: continue
        v=[]
        for t in range(10):
            Dp=D[g.permutation(n)]; Bp=fit(Xz,Dp,dev)
            v.append(float(correct(fas(Bp),E).mean()))
        rec(test="control",ref=name,mean=float(np.mean(v)),sd=float(np.std(v)),reps=10)
    print("DONE_REAL4",flush=True)
