#%%
import numpy as np
from matplotlib import pyplot as plt
n1 = np.random.normal(0, 1, 1000)
plt.hist(n1)

#%%
n2 = np.random.uniform(0, 1, 1000)
plt.hist(n2)
#%%spring values changing over time
Kts_c = 0.1; Kts_m = 0.1; ts = np.arange(100)
Kts = Kts_c + Kts_m*ts + np.random.normal(0, 1, 100)
plt.plot(tss,Kts); plt.show()
#%%Force changing over time
Fts_c = 0.5; Fts_m = 0.1; ts = np.arange(100)
Fts = Fts_c + Fts_m*ts + np.random.normal(0, 1, 100)
plt.plot(tss,Fts); plt.show()
#%%Spring damper model where spring force is changing with time
#c = 10 #Damping constant
#K = 1 #Spring constant
m = 4 #mass
F = 0.5 #Force

Ts = 0.1
Tstart = 0
Tstop = 10000
N = int((Tstop -Tstart)/Ts); print(N)
x1 = np.zeros(N+2);
x2 = np.zeros(N+2);
x1[0] = 0.3; print(x1[0])
x2[0] = 0

Kts_c = 0.1; Kts_m = 0.0001; Tss = np.arange(N+2)
Kts = kts_c + Kts_m*Tss + np.random.normal(0, 1, (N+2))
Fts_c = 3; Fts_m = 0.0005; Tss = np.arange(N+2);
Fts = Fts_c + Fts_m*Tss + np.random.normal(0, 1, (N+2))
Cts_c = 5; Cts_m = 0.0001; Tss = np.arange(N+2);
Cts = Cts_c + Cts_m*Tss + np.random.normal(0, 1, (N+2))
Mts = m + 0.00*Tss + np.random.uniform(-3, 3, (N+2))
print(len(x1)); print(len(Fts)); print(len(Kts));
tss = np.arange(N+2); 
plt.plot(tss, Fts); 
#plt.plot(tss, Kts); plt.plot(tss, Cts); plt.plot(Tss, Mts)
#%%
print(np.average(Fts)); print(np.average(Cts)); print(np.average(Kts)); print(np.average(Mts))
#%%

for k in np.arange(N+1):
    a11 = 1; a12 = Ts; 
    a21 = -(Kts[k]*Ts)/Mts[k]; a22 = 1 - ((Ts*Cts[k])/Mts[k])
    b11 = 0; b21 = Ts/Mts[k]
    x1[k+1] = a11*x1[k] + a12*x2[k] + b11*Fts[k]
    x2[k+1] = a21*x1[k] + a22*x2[k] + b21*Fts[k]
    #print(k)
#%%
print(x1[10000])
#%%plotting
t = np.arange(Tstart, Tstop+2*Ts, Ts)
plt.plot(t,x1)
plt.plot(t,x2)
plt.title("Simulation of spring-damper system")
plt.xlabel('time'); plt.ylabel('state variable')
#plt.grid(); plt.show()