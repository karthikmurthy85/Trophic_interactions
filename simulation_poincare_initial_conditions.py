#!/usr/bin/env python
# coding: utf-8

# In[2]:


import sympy as sm
import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import timeit


# In[3]:


rr = 0.83; aa1 = 1.66; bb1 = 0.33; d1 = 0.4
aa2 = 0.05; bb2 = 0.5; d2 = 0.01


# In[4]:


##Hastings to MacArthur model conversion##
r0 = rr; mC0 = d1; mP0 = d2; ##Easy ones

q0 = 1/10000; aP0 = 0.01; eP0 = 0.04 ##Random allocation

aC0 = ((aa1/bb1) * (bb2/aa2))*aP0*eP0
aC0

hC0 = q0/(bb1*aC0)
hC0

eC0 = hC0*aa1
eC0

print(aa1/bb1)
print((eC0*aC0)/q0)

hP0 = q0/(bb2*aP0*eC0)
hP0

eP0 = aa2*hP0
eP0


# In[5]:


u0 = np.array([0.8, 0.61, 9.742, 0])
v0 = u0
v0[0] = u0[0]/q0
v0[1] = u0[1]*eC0/q0
v0[2] = u0[2]*eC0*eP0/q0
v0


# In[28]:


v00 = v0*np.random.normal(loc=1, scale=1, size=(200, 4)) + v0
v00=v00[np.where(v00[:,0] > 0)]
v00=v00[np.where(v00[:,1] > 0)]
v00=v00[np.where(v00[:,2] > 0)]


# In[31]:


p = (r0, q0, aC0, eC0, hC0, mC0, aP0, eP0, mP0, hP0)
p


# In[32]:


Xr, X0, Ex0, K, T = sm.symbols('X_r, X_0, E_x0, K, T ')
#Xr = X0*sm.exp(-Ex0/(K*T))
fx = Xr - X0*sm.exp(-Ex0/(K*T))
fx


# In[33]:


##Temperature for temperate case##
tmp = np.linspace(297.1, 298.35, 10); 
tmp[0] ##Initial temperature 
T0 = tmp[0]


# In[34]:


##Intrinsic growth rate of resources
fx1 = fx.subs([(Ex0, 1.60218e-19 * 0.5), (K, 1.380649e-23), (T, T0), (Xr, r0)]) 
xr = sm.solve(fx1, X0)

fxT = X0*sm.exp(-Ex0/(K*T))
frT1 = fxT.subs([(Ex0, 1.60218e-19 * 0.5), (K, 1.380649e-23), (X0, xr[0])])
frT2 = sm.lambdify(T, frT1, 'numpy')
print(frT2(285))
print(frT2(297))
print(frT2(300))


# In[35]:


##Carrying capacity of resources##
K0 = 1/q0
fx1 = fx.subs([(Ex0, 1.60218e-19 * 0.1), (K, 1.380649e-23), (T, T0), (Xr, K0)]) 
xK = sm.solve(fx1, X0)

fxT = X0*sm.exp(-Ex0/(K*T))
fKT1 = fxT.subs([(Ex0, 1.60218e-19 * 0.1), (K, 1.380649e-23), (X0, xK[0])])
fqT2 = (sm.lambdify(T, 1/fKT1, 'numpy'))
print(fqT2(285))
print(fqT2(297))
print(fqT2(300))


# In[36]:


##Attack rate of consumers

fx1 = fx.subs([(Ex0, 1.60218e-19 * 0.7), (K, 1.380649e-23), (T, T0), (Xr, aC0)]) 
xaC = sm.solve(fx1, X0)

fxT = X0*sm.exp(-Ex0/(K*T))
faCT1 = fxT.subs([(Ex0, 1.60218e-19 * 0.7), (K, 1.380649e-23), (X0, xaC[0])])
faCT2 = sm.lambdify(T, faCT1, 'numpy')
print(faCT2(285))
print(faCT2(297))
print(faCT2(300))


# In[37]:


##Efficiency of consumers##

fx1 = fx.subs([(Ex0, 1.60218e-19 * 0.6), (K, 1.380649e-23), (T, T0), (Xr, eC0)]) 
xeC = sm.solve(fx1, X0)

fxT = X0*sm.exp(-Ex0/(K*T))
feCT1 = fxT.subs([(Ex0, 1.60218e-19 * 0.6), (K, 1.380649e-23), (X0, xeC[0])])
feCT2 = sm.lambdify(T, feCT1, 'numpy')
print(feCT2(285))
print(feCT2(297))
print(feCT2(305))


# In[38]:


##Handling cost - consumers##

fx1 = fx.subs([(Ex0, 1.60218e-19 * 0.6), (K, 1.380649e-23), (T, T0), (Xr, hC0)]) 
xhC = sm.solve(fx1, X0)

fxT = X0*sm.exp(-Ex0/(K*T))
fhCT1 = fxT.subs([(Ex0, 1.60218e-19 * 0.6), (K, 1.380649e-23), (X0, xhC[0])])
fhCT2 = sm.lambdify(T, fhCT1, 'numpy')
print(fhCT2(285))
print(fhCT2(297))
print(fhCT2(300))


# In[39]:


#Intrinsic mortality rate of consumers##

fx1 = fx.subs([(Ex0, 1.60218e-19 * 0.6), (K, 1.380649e-23), (T, T0), (Xr, mC0)]) 
xmC = sm.solve(fx1, X0)

fxT = X0*sm.exp(-Ex0/(K*T))
fmCT1 = fxT.subs([(Ex0, 1.60218e-19 * 0.6), (K, 1.380649e-23), (X0, xmC[0])])
fmCT2 = sm.lambdify(T, fmCT1, 'numpy')
print(fmCT2(285))
print(fmCT2(297))
print(fmCT2(300))


# In[40]:


##Attack rate of predators
fx1 = fx.subs([(Ex0, 1.60218e-19 * 0.4), (K, 1.380649e-23), (T, T0), (Xr, aP0)]) 
xaP = sm.solve(fx1, X0)

fxT = X0*sm.exp(-Ex0/(K*T))
faPT1 = fxT.subs([(Ex0, 1.60218e-19 * 0.4), (K, 1.380649e-23), (X0, xaP[0])])
faPT2 = sm.lambdify(T, faPT1, 'numpy')
print(faPT2(285))
print(faPT2(297))
print(faPT2(300))


# In[41]:


##Efficiency of predators
fx1 = fx.subs([(Ex0, 1.60218e-19 * 0.4), (K, 1.380649e-23), (T, T0), (Xr, eP0)]) 
xeP = sm.solve(fx1, X0)

fxT = X0*sm.exp(-Ex0/(K*T))
fePT1 = fxT.subs([(Ex0, 1.60218e-19 * 0.4), (K, 1.380649e-23), (X0, xeP[0])])
fePT2 = sm.lambdify(T, fePT1, 'numpy')
print(fePT2(285))
print(fePT2(297))
print(fePT2(300))


# In[42]:


##Intrisic mortality rate of predators#
fx1 = fx.subs([(Ex0, 1.60218e-19 * 0.4), (K, 1.380649e-23), (T, T0), (Xr, mP0)]) 
xmP = sm.solve(fx1, X0)

fxT = X0*sm.exp(-Ex0/(K*T))
fmPT1 = fxT.subs([(Ex0, 1.60218e-19 * 0.4), (K, 1.380649e-23), (X0, xmP[0])])
fmPT2 = sm.lambdify(T, fmPT1, 'numpy')
print(fmPT2(285))
print(fmPT2(297))
print(fmPT2(300))


# In[43]:


##handling cost - Predators #
fx1 = fx.subs([(Ex0, 1.60218e-19 * 0.4), (K, 1.380649e-23), (T, T0), (Xr, hP0)]) 
xhP = sm.solve(fx1, X0)

fxT = X0*sm.exp(-Ex0/(K*T))
fhPT1 = fxT.subs([(Ex0, 1.60218e-19 * 0.4), (K, 1.380649e-23), (X0, xhP[0])])
fhPT2 = sm.lambdify(T, fhPT1, 'numpy')
print(fhPT2(285))
print(fhPT2(297))
print(fhPT2(300))


# In[44]:


def RCP_temp (t, u, ct, mt):

    Tt= ct + mt*t
    
    r1 = np.around(frT2(Tt),6); q1 = np.around(fqT2(Tt),10)
    aC1 = np.around(faCT2(Tt),6); eC1 = np.around(feCT2(Tt),6); hC1 = np.around(fhCT2(Tt),6); mC1 = np.around(fmCT2(Tt),6); 
    aP1 = np.around(faPT2(Tt),6); eP1 = np.around(fePT2(Tt),6); hP1 = np.around(fhPT2(Tt),6); mP1 = np.around(fmPT2(Tt),6);
  
    
    p0 = (ct, Tt, r1, q1, aC1, eC1, hC1, mC1, aP1, eP1, mP1, hP1)
    #print(p0)
    
    du = np.zeros([4, 1])
    
    #variables
    R = u[0]
    C = u[1]
    P = u[2]
    
    du[0] = r1*R*(1 - (R*q1)) - (aC1*R*C)/(1 + (aC1*hC1*R))
    du[1] = (eC1*aC1*R*C)/(1 + (aC1*hC1*R)) - (aP1*C*P)/(1+ (aP1*hP1*C)) - mC1*C
    du[2] = (eP1*aP1*C*P)/(1+ (aP1*hP1*C)) - mP1*P
    du[3] = 1
    
    
    
    du = du.reshape(4,)
    
    return(du)


# In[ ]:


def poincare_Rplane(dfsol, thrsh):
    
    import scipy.interpolate

    up_entr = np.zeros((1,4))

    for i in (np.arange(np.shape(dfsol)[1])-1):

        x1 = dfsol[0][i] - thrsh; x2 = dfsol[0][i+1] - thrsh

        if ((x1 > 0) and (x1 * x2) < 0):

            y_interp = scipy.interpolate.interp1d([ dfsol[0][i],  dfsol[0][i+1]], [dfsol[1][i], dfsol[1][i+1]])
            ypcr = y_interp(thrsh)

            z_interp = scipy.interpolate.interp1d([ dfsol[0][i],  dfsol[0][i+1]], [dfsol[2][i], dfsol[2][i+1]])
            zpcr = z_interp(thrsh)

            t_interp = scipy.interpolate.interp1d([ dfsol[0][i],  dfsol[0][i+1]], [dfsol[3][i], dfsol[3][i+1]])
            tpcr = t_interp(thrsh)

            Tmp= tmp[0] + tr*tpcr

            up_entr1 = np.array([ypcr, zpcr, tpcr, Tmp]).reshape(1,4)
            up_entr = np.concatenate([up_entr, up_entr1], axis = 0)
            
    return(up_entr)


# In[45]:


tdif = tmp[len(tmp)-1] - tmp[0]

tend = 10000
t_span = (0, tend)
tint = np.linspace(0, tend, tend*10)
    
tr = tdif/tend;

p = (tmp[0], tr)


# In[55]:


def poincare_simu(i):
    
    sol1 = solve_ivp (RCP_temp, t_span, v00[i], args = p, t_eval = tint, method= 'RK45', rtol=1e-12, atol=1e-12)
    
    ssv0 = sol1.y[:,10:]
    tm = ssv0[3]; R0 = ssv0[0]; C0 = ssv0[1]; P0 = ssv0[2]
    thrsh1 = np.max(R0)*0.75 ##for poincare section in C,P plane
    dfsol1 = sol1.y
    
    updf1 = poincare_Rplane(dfsol1, thrsh1)
    
    updf = np.delete(updf1, [0,1], axis=0)
    
    id="%03d" % i
    fnm = "poincare_section_tropical" + id + ".csv"
    df.to_csv(fnm)


# In[67]:


Parallel(n_jobs=4)(delayed(poincare_simu)(i) for i in np.arange(100))

