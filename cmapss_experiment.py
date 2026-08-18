import os, json, math, random, urllib.request
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

BASE='https://raw.githubusercontent.com/hankroark/Turbofan-Engine-Degradation/master/CMAPSSData'
OUT=Path('cmapss_results'); OUT.mkdir(exist_ok=True)
DATA=Path('cmapss_data'); DATA.mkdir(exist_ok=True)
COLS=['unit','cycle']+[f'op{i}' for i in range(1,4)]+[f's{i}' for i in range(1,22)]
SENSORS=['s2','s3','s4','s7','s8','s9','s11','s12','s13','s14','s15','s17','s20','s21']


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def download(name):
    p=DATA/name
    if not p.exists(): urllib.request.urlretrieve(f'{BASE}/{name}', p)
    return p


def read_train(fd):
    p=download(f'train_{fd}.txt')
    return pd.read_csv(p, sep=r'\s+', header=None, names=COLS, engine='python')


def split_units(df, seed=2026):
    units=np.array(sorted(df.unit.unique()))
    rng=np.random.default_rng(seed); rng.shuffle(units)
    n=len(units); ntr=int(n*.70); nva=int(n*.15)
    return set(units[:ntr]), set(units[ntr:ntr+nva]), set(units[ntr+nva:])


def build_hi(df, train_units):
    d=df.copy()
    # six operating conditions from the 3 settings, fitted only on training engines
    km=KMeans(n_clusters=6, random_state=2026, n_init=20)
    km.fit(d[d.unit.isin(train_units)][['op1','op2','op3']])
    d['cond']=km.predict(d[['op1','op2','op3']])
    # condition-specific healthy baseline using first 20% (at least 15, at most 40 cycles) of TRAIN engines
    healthy=[]
    for u,g in d[d.unit.isin(train_units)].groupby('unit'):
        n=max(15,min(40,int(math.ceil(len(g)*.20))))
        healthy.append(g.sort_values('cycle').head(n))
    healthy=pd.concat(healthy,ignore_index=True)
    stats={}
    global_mu=healthy[SENSORS].mean(); global_sd=healthy[SENSORS].std().replace(0,1.0).fillna(1.0)
    for c in range(6):
        z=healthy[healthy.cond==c]
        if len(z)<20: stats[c]=(global_mu,global_sd)
        else: stats[c]=(z[SENSORS].mean(),z[SENSORS].std().replace(0,1.0).fillna(1.0))
    scores=[]
    for c,g in d.groupby('cond'):
        mu,sd=stats[int(c)]
        z=((g[SENSORS]-mu)/sd).abs().clip(0,10)
        # robust degradation magnitude; median prevents a single noisy sensor dominating
        sc=z.median(axis=1)
        scores.append(pd.Series(sc.values,index=g.index))
    score=pd.concat(scores).sort_index()
    # map baseline-normalized deviation to 0-100 HI; eta fixed from training healthy/degradation distribution
    eta=float(np.quantile(score[d.unit.isin(train_units)],0.90))
    eta=max(eta,0.25)
    d['hi']=100*np.exp(-score/eta)
    # small causal smoothing only, same for all variants
    d['hi']=d.groupby('unit')['hi'].transform(lambda s:s.ewm(span=5,adjust=False).mean())
    return d, {'eta':eta,'centers':km.cluster_centers_.tolist()}


class WinDS(Dataset):
    def __init__(self, df, units, L, K, scaler):
        self.items=[]
        for u,g in df[df.unit.isin(units)].groupby('unit'):
            g=g.sort_values('cycle')
            ops=scaler.transform(g[['op1','op2','op3']])
            hi=g.hi.to_numpy(dtype=np.float32)/100.0
            for t in range(L-1,len(g)-K):
                hist=np.column_stack([ops[t-L+1:t+1],hi[t-L+1:t+1]])
                fut_ops=ops[t+1:t+K+1]
                fut_hi=hi[t+1:t+K+1]
                h0=hi[t]
                self.items.append((hist.astype(np.float32),fut_ops.astype(np.float32),fut_hi.astype(np.float32),np.float32(h0)))
    def __len__(self): return len(self.items)
    def __getitem__(self,i): return self.items[i]


class Model(nn.Module):
    def __init__(self, hidden=32):
        super().__init__()
        self.enc=nn.GRU(4,hidden,batch_first=True)
        self.dec=nn.GRU(3,hidden,batch_first=True)
        self.head=nn.Linear(hidden,1)
    def forward(self,hist,fops):
        _,h=self.enc(hist)
        z,_=self.dec(fops,h)
        return self.head(z).squeeze(-1)


def losses(kind, raw, y, h0):
    true_inc=torch.cat([y[:,:1]-h0[:,None],y[:,1:]-y[:,:-1]],1)
    true_cum=y-h0[:,None]
    if kind=='B':
        pred=raw; loss=((pred-y)**2).mean()
    elif kind=='C':
        inc=raw; pred=h0[:,None]+torch.cumsum(inc,1); loss=((inc-true_inc)**2).mean()
    elif kind=='D':
        cum=raw; pred=h0[:,None]+cum; loss=((cum-true_cum)**2).mean()
    elif kind=='E':
        inc=raw; pred=h0[:,None]+torch.cumsum(inc,1)
        loss=((inc-true_inc)**2).mean()+((pred-y)**2).mean()
    return loss,pred


def evaluate(model, loader, kind, device):
    model.eval(); ys=[]; ps=[]
    with torch.no_grad():
        for hist,fops,y,h0 in loader:
            hist,fops,y,h0=[x.to(device) for x in (hist,fops,y,h0)]
            raw=model(hist,fops); _,p=losses(kind,raw,y,h0)
            ys.append(y.cpu().numpy()); ps.append(p.cpu().numpy())
    y=np.concatenate(ys); p=np.concatenate(ps)
    e=(p-y)*100
    return {'MAE':float(np.mean(np.abs(e))),'RMSE':float(np.sqrt(np.mean(e**2))),'LAST_MAE':float(np.mean(np.abs(e[:,-1])))}


def train_one(df,tr,va,te,L,K,kind,seed,opsc,device):
    seed_all(seed)
    train=WinDS(df,tr,L,K,opsc); val=WinDS(df,va,L,K,opsc); test=WinDS(df,te,L,K,opsc)
    gen=torch.Generator().manual_seed(seed)
    dl=DataLoader(train,256,shuffle=True,generator=gen); vl=DataLoader(val,512); tl=DataLoader(test,512)
    model=Model(32).to(device); opt=torch.optim.Adam(model.parameters(),lr=1e-3,weight_decay=1e-5)
    best=1e99; state=None; bad=0
    for ep in range(40):
        model.train()
        for hist,fops,y,h0 in dl:
            hist,fops,y,h0=[x.to(device) for x in (hist,fops,y,h0)]
            opt.zero_grad(); raw=model(hist,fops); loss,_=losses(kind,raw,y,h0); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
        model.eval(); vals=[]
        with torch.no_grad():
            for hist,fops,y,h0 in vl:
                hist,fops,y,h0=[x.to(device) for x in (hist,fops,y,h0)]
                raw=model(hist,fops); loss,_=losses(kind,raw,y,h0); vals.append(loss.item())
        v=float(np.mean(vals))
        if v<best-1e-6: best=v; state={k:v.cpu().clone() for k,v in model.state_dict().items()}; bad=0
        else: bad+=1
        if bad>=6: break
    model.load_state_dict(state)
    m=evaluate(model,tl,kind,device); m['epochs']=ep+1; m['n_train']=len(train); m['n_val']=len(val); m['n_test']=len(test)
    return m


def main():
    torch.set_num_threads(2)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    rows=[]; meta={}
    for fd in ['FD002','FD004']:
        raw=read_train(fd); tr,va,te=split_units(raw)
        df,hmeta=build_hi(raw,tr)
        opsc=StandardScaler().fit(df[df.unit.isin(tr)][['op1','op2','op3']])
        meta[fd]={'n_rows':len(raw),'n_units':raw.unit.nunique(),'train_units':len(tr),'val_units':len(va),'test_units':len(te),'hi_meta':hmeta}
        for K in [3,5,10]:
            for kind in ['B','C','D','E']:
                for seed in [11,22,33]:
                    m=train_one(df,tr,va,te,20,K,kind,seed,opsc,device)
                    row={'dataset':fd,'K':K,'model':kind,'seed':seed,**m}; rows.append(row)
                    print(row,flush=True)
    res=pd.DataFrame(rows); res.to_csv(OUT/'all_runs.csv',index=False)
    summary=res.groupby(['dataset','K','model'])[['MAE','RMSE','LAST_MAE']].agg(['mean','std']).reset_index()
    summary.columns=['_'.join([str(x) for x in c if x!='']).rstrip('_') if isinstance(c,tuple) else c for c in summary.columns]
    summary.to_csv(OUT/'summary.csv',index=False)
    with open(OUT/'meta.json','w') as f: json.dump(meta,f,indent=2)
    print('\nSUMMARY\n',summary.to_string(index=False))

if __name__=='__main__': main()
