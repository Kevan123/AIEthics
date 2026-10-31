import itertools, numpy as np
from scipy.optimize import linprog
pats=list(itertools.product([0,1],repeat=3))
def lp_bounds(e,r):
    # general (non-exchangeable) joint over 8 patterns with all marginals = e and all pairs = r
    A=[np.ones(8)];b=[1]
    for i in range(3): A.append([p[i] for p in pats]);b.append(e)
    for i,j in itertools.combinations(range(3),2): A.append([p[i]*p[j] for p in pats]);b.append(r)
    c=np.array([float(p==(1,1,1)) for p in pats])
    lo=linprog(c,A_eq=A,b_eq=b,bounds=(0,None),method='highs'); hi=linprog(-c,A_eq=A,b_eq=b,bounds=(0,None),method='highs')
    return lo.fun,-hi.fun
for e,r in [(0.4,0.2),(0.15,0.0225),(0.15,0.10),(0.3,0.05),(0.55,0.4)]:
    cf=(max(0,2*r-e),min(r,1-3*e+3*r))
    print(e,r,'LP',np.round(lp_bounds(e,r),5),'closed',np.round(cf,5),'indep e^3',round(e**3,5))
# Prop 1 check
for name,pk in [('I',{0:.2,2:.6,3:.2}),('II',{1:.6,3:.4})]:
    acc=sum(k*p for k,p in pk.items())/3; pair=sum(p*(k*(k-1)/6) for k,p in pk.items()); maj=sum(p for k,p in pk.items() if k>=2)
    pe=1-2*acc+pair; corr=(pe-(1-acc)**2)/((1-acc)*acc)
    print(name,round(acc,4),round(pair,4),round(corr,4),round(maj,4))
