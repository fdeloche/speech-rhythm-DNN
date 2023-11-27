#Testing distribution of pearson coefficient r if random

import os
os.chdir("/Users/violette_dau/stage")
import scipy.integrate as scpi
import math
import numpy as np
import matplotlib.pyplot as plt

def fB(t,x,y):
    f=t**(x-1)
    f=f*((1-t)**(y-1))
    return f

def Beta(x,y):
    return scpi.quad(fB,0,1,args=(x,y))[0]

def f(r,n):
    num=(1-r**2)**((n-4)/2)
    den=Beta(1/2,(n-2)/2)
    return num/den
    
x=np.linspace(-1,1,50)
y=[]
for i in x:
    y.append(f(math.sqrt(i**2),153))
plt.plot(x,y)
plt.title('Densité de répartition des coefficients de corrélation r pour 153 points')
plt.savefig('distribpearson.pdf')
plt.close()

print(1-(1-scpi.quad(f,0.4,1,args=153)[0]*2)**512)