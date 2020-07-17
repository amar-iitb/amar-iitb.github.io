#!/usr/bin/env python
# coding: utf-8

# In[23]:


# %load Morse\ potential\ split.py

from pylab import *
import numpy as np
import scipy.linalg as la
import matplotlib as plt



# In[2]:


def wavefunction_norm(x, psi):
    """
    Calculate the norm of the given wavefunction.
    """
    dx = x[1] - x[0]
    return (psi.conj() * psi).sum() * dx


def wavefunction_normalize(x, psi):
    """
    Normalize the given wavefunction.
    """
    return psi / np.sqrt(wavefunction_norm(x, psi))


# In[22]:


def w(t):
        return 0.
        #return (0.1*x*sin(0.2*t)*(sin(pi*t))**2)


# In[4]:


# In[ ]:


def v(x):
	x=asarray(x)
	y=zeros(x.shape)
	new=x[x>-5]
	y[x>-5]=0.103*(1-exp(-0.72*(new-2.0)))**2   
	y[x>5]=0              
	return y


# In[38]:


def V(x):
    return 0.103*(1-exp(-0.72*(x-2.0)))**2 


N=1001
x0,x1=0.5,100.    #box limits
x=linspace(x0,x1,N)
dx=(x1-x0)/N
dt=0.5
U=V(x)
plot(x,U)
# In[6]:


# In[39]:


#Hamiltonian

D=(diag(ones(N-1),-1)-diag(2*ones(N))+diag(ones(N-1),+1))/dx**2
M=diag(ones(N))+(dx**2/12)*D
h=dot(inv(M),D)
H=-0.5*h+diag(V(x))

#H=-0.5*(diag(ones(N-1),-1)-diag(2.*ones(N))+diag(ones(N-1),+1))/dx**2 +diag(v(x))


# In[7]:


val,vec=la.eigh(H)
vec_inv=vec.conj().transpose()




# In[8]:



for i in range(0,2):
    plot(x,wavefunction_normalize(x,vec[:,i]))
show()
   


# In[9]:


val[0:10]


# In[40]:



fig, ax = subplots()

ax.plot(x, U, 'k')
for n in range(2):
    Y =vec[:,n] + val[n]
    Y1= val[n]
    mask = where(Y1 > U) 
    ax.plot(x[mask], val[n].real * ones(shape(x))[mask], 'k--')
    mask =where (Y > U - 2.0)
   # print(mask)
    ax.plot(x[mask],Y[mask].real)
   
    
ax.set_xlim(0, 10)
ax.set_ylim(0, 0.2)
ax.set_xlabel(r'$x$', fontsize=18)
ax.set_ylabel(r'$U(x)$', fontsize=18);


# In[ ]:


# In[29]:


#initial wavefunction
Psi=exp(-((x)**2)*(0.2/2) + 1j*x)
#+exp(-x**2/2 + 1j*x)
#Psi=vec[:,0]
Psi0=Psi
Psi=wavefunction_normalize(x,Psi)          #normalise
print((Psi.conj() * Psi).sum() * dx)       #Inner Product
Psi.shape


# In[30]:


plot(x,Psi.real)


# In[31]:


# In[ ]:


"""
inpr=[]
r=0.
r+=(wavefunction_normalize(x,vec[:,0]).conj()*vec[:,0]).sum()*dx
inpr.append(r)
print(inpr)
"""

vec_norm=wavefunction_normalize(x,vec[:,1])
print(vec_norm.shape)
print(abs((vec_norm*Psi).sum()*dx))


# In[32]:


# In[ ]:


#time loop
p=[Psi]
inpr=[]
t=0.
for n in range(0,720):
    r=0.
    r+=abs((vec_norm*Psi).sum()*dx)
    inpr.append(r)
    Psi*=exp(-0.5*1j*dt*(w(t)))
    Psi=dot(vec_inv,Psi)
    Psi*=exp(-1j*dt*val)
    Psi=dot(vec,Psi)
    Psi*=exp(-0.5*1j*dt*(w(t)))
    p.append(Psi)
    t+=dt
    


# In[ ]:


# In[33]:


plot(x,p[0].real**2+p[0].imag**2,label='0')
plot(x,p[5].real**2+p[5].imag**2,label='5')
plot(x,p[100].real**2+p[100].imag**2.,label='100')
plot(x,Psi.real**2+Psi.imag**2,label='final')
legend()
show()


# In[34]:


# In[ ]:


ttt=linspace(0,720*dt,720)
inner=np.array(inpr)
step(ttt,abs(inner))
#savefig("Perturbation t to HO at dt=0.5.png", dpi=200)
show()


# In[118]:


plot(x,Psi.real,label=r"$\Psi_{real}$")
plot(x,Psi.imag,label=r"$\Psi_{imag}$")
plot(x,Psi0.conj().real,'g--',label=r"$\Psi0_{real}$")
plot(x,Psi0.conj().imag,'y--',label=r"${\Psi_{0}}_{imag}$")
legend()
plt.show()


# In[3]:


from matplotlib import pyplot as plt 
import numpy as np 
from matplotlib.animation import FuncAnimation 

# initializing a figure in 
# which the graph will be plotted 
fig = plt.figure() 

# marking the x-axis and y-axis 
axis = plt.axes(xlim =(-50,50), ylim =(-0.5, 1)) 
plt.plot(x,v(x)*0.1)
# initializing a line variable 
line = axis.plot(x, p[0].real**2+p[0].imag**2, lw = 2)[0] 

# data which the line will 
# contain (x, y) 
def init(): 
	line.set_data([], []) 
	return line 

def animate(i): 
    # plots a sine graph 
    line.set_ydata(p[i].real**2+p[i].imag**2)


    axis.set_ylabel('time(s):'+str(dt*i))
    axis.set_title('Frame'+str(i))

    return line 
anim = FuncAnimation(
    fig, animate, interval=70, frames=700)
anim.save('Gaussian Prop.mp4')

plt.draw()
plt.show()
 


# In[ ]:




