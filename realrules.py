import csv,json,os
import numpy as np,torch
from real_ref import zscore,load_edges,edge_cons,scaled
import oriented as O
R,L=20,1e-2
def P_(B):
    A=B.abs();d=A.diagonal().clone();q=d[d>0]
    md=q.median() if q.numel() else torch.tensor(1.,device=B.device,dtype=B.dtype)
    d=torch.where(d>1e-12*md,d,md);C=A/d[None,:];C.fill_diagonal_(0.);return C,C.T
def score(B):
    C,_=P_(B);return torch.argsort((C-C.T).sum(1))
def copeland(B):
    _,P=P_(B);return torch.argsort(-torch.sign(P-P.T).sum(1))
def fas(B):
    _,P=P_(B);P=P.cpu().numpy();p=P.shape[0];l=[];r=[];rem=list(range(p))
    while rem:
        i=np.array(rem);s=P[np.ix_(i,i)];o=s.sum(1);n=s.sum(0)
        k=np.where(o<=0)[0]
        if k.size:v=int(i[k[0]]);r.append(v);rem.remove(v);continue
        k=np.where(n<=0)[0]
        if k.size:v=int(i[k[0]]);l.append(v);rem.remove(v);continue
        v=int(i[int(np.argmax(o-n))]);l.append(v);rem.remove(v)
    return torch.tensor(l+r[::-1])
RU={"score":score,"peel":O.order_from_B,"copeland":copeland,"fas":fas}
def fit(Xz,D,dev):
    return O.fit_B(torch.from_numpy(Xz.astype(np.float32)).to(dev),
                   torch.from_numpy(np.nan_to_num(D).astype(np.float32)).to(dev),lam=L)
if __name__=="__main__":
    g=np.random.default_rng(0)
    z=np.load("cache/panel2000_depvar.npz",allow_pickle=True)
    sym=list(map(str,z["symbols"]));Xr=np.asarray(z["X_s"],dtype=np.float64)
    Ds=np.asarray(z["D_s"],dtype=np.float64)
    E=load_edges("ref/trrust_rawdata.human.tsv",sym);Xz=zscore(Xr)
    dev="cuda" if torch.cuda.is_available() else "cpu";rows=[]
    for mode in ("rescaled","raw"):
        D=scaled(Ds,Xr,1.0,mode,g)
        B=fit(Xz,D,dev);Bp=[fit(Xz,D[g.permutation(D.shape[0])],dev) for _ in range(R)]
        for nm,f in RU.items():
            v=edge_cons(f(B),E);c=[edge_cons(f(b),E) for b in Bp]
            rec=dict(mode=mode,rule=nm,edge_cons=float(v),ctrl=float(np.mean(c)),
                     ctrl_sd=float(np.std(c)),margin=float(v-np.mean(c)),reps=R,n_edges=len(E))
            rows.append(rec);print(json.dumps(rec),flush=True)
    os.makedirs("rules_out",exist_ok=True)
    k=sorted({x for r in rows for x in r})
    with open("rules_out/realrules.csv","w",newline="") as fh:
        w=csv.DictWriter(fh,fieldnames=k);w.writeheader();w.writerows(rows)
    print("DONE_RULES")
