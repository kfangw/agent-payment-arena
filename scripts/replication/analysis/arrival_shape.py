"""Stage-0 activation exposure and settled threshold under a uniform vs geometric answer arrival.
Same answer rate q, same deadline tau; the compiler assumes geometric, the world is uniform."""
import sys, os, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import model as M

def window_AD_general(f, sig, v, m, cw, lam, tau):
    # lam[t] = arrival hazard at tick t (t=1..tau), tick order: pay cw, test arrival, transition
    N=len(f)-1; FIN=N+1
    ex=np.maximum(np.array([v*(sig[j]*(1+m)-1.0) for j in range(N+2)]),0.0)
    A=np.zeros((N+2,tau+1)); D=np.zeros((N+2,tau+1))
    # t = ticks remaining; tick index k = tau - t + 1
    for t in range(1,tau+1):
        k=tau-t+1; rho=lam[k]
        for j in range(FIN,-1,-1):
            if j==FIN: cA,cD,surv=A[FIN,t-1],D[FIN,t-1],1.0
            else: cA,cD,surv=A[j+1,t-1],D[j+1,t-1],1.0-f[j]
            A[j,t]=rho*ex[j]+(1-rho)*surv*cA
            D[j,t]=cw*v+(1-rho)*surv*cD
    return A[:,tau],D[:,tau]

def verify_active_stage0(f,v,p,lam,pis):
    m,h,C,cw,tau=p['m'],p['h'],p['C'],p['cw'],p['tau']
    sig=M.sigma_list(np.array(f)); A,D=window_AD_general(f,sig,v,m,cw,lam,tau)
    N=len(f)-1; FIN=N+1
    G=np.array([v*(sig[i]*(1+(1-pis)*m-pis*h)-1.0) for i in range(N+2)])
    W=np.array([-C-D[i]+(1-pis)*A[i] for i in range(N+2)])
    V=np.zeros_like(G); V[FIN]=np.maximum.reduce([G[FIN],0*G[FIN],W[FIN]])
    Wait=np.full_like(G,-np.inf)
    for i in range(N,-1,-1):
        Wait[i]=-cw*v+(1-f[i])*V[i+1]; V[i]=np.maximum.reduce([G[i],0*G[i],W[i],Wait[i]])
    return (W[0]>=np.maximum.reduce([G[0],0*G[0],Wait[0]])-1e-12).any(), (W[FIN]>=np.maximum(G[FIN],0)-1e-12).any()

def activation(f,p,lam,which):
    pis=np.linspace(0,1,2001)
    lo,hi=0.05,500.0
    for _ in range(60):
        mid=np.sqrt(lo*hi)
        s0,sF=verify_active_stage0(f,mid,p,lam,pis)
        if (s0 if which==0 else sF): hi=mid
        else: lo=mid
    return hi

def hazards(kind,q,tau):
    if kind=='geo':
        rho=1-(1-q)**(1/tau); return {t:rho for t in range(1,tau+1)}
    a=q/tau; lam={}; rem=1.0
    for t in range(1,tau+1): lam[t]=a/rem; rem-=a
    return lam

q=0.75
cases={'E-outage chain (tau=10)':(np.array([0.005]+[3e-5]*17),dict(m=0.35,h=1.0,C=0.5,cw=6e-3,tau=10)),
       'E-slow chain (tau=300)':(np.array([0.06*0.5**i for i in range(8)]),dict(m=0.35,h=1.0,C=0.5,cw=2e-4,tau=300)),
       'E-fast chain (tau=300)':(np.array([0.005]+[1e-6]*40),dict(m=0.35,h=1.0,C=0.5,cw=2e-4,tau=300))}
for name,(f,p) in cases.items():
    out={}
    for kind in ['geo','uni']:
        lam=hazards(kind,q,p['tau'])
        out[kind]=(activation(f,p,lam,0),activation(f,p,lam,1))
    g,u=out['geo'],out['uni']
    print(f"{name}: stage-0 activation exposure geo {g[0]:.3f} uni {u[0]:.3f} -> {100*(g[0]/u[0]-1):+.1f}% ; settled threshold geo {g[1]:.3f} uni {u[1]:.3f} -> {100*(g[1]/u[1]-1):+.1f}%")
