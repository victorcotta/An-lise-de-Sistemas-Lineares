from cmath import exp
from re import T
from scipy.integrate import solve_ivp
from scipy.integrate import quad
from scipy.misc import derivative
import matplotlib.pyplot as plt
import numpy as np
import math as m

plt.close('all')
 
#VARIÁVEIS DE SIMULAÇÃO 
T  = 0 #TEMPO REAL 
Ti = 0 #SEGUNDOS
Tf = 40 #SEGUNDOS
step = 0.01 #SEGUNDOS

#VARIÁVEIS DO SISTEMA 
I=0 #CORRENTE INICIAL
V=0

#EIXOS DE PLOTAGEM
I_eixo = []
V_eixo = []
T_eixo = []

while(T<=Tf):
    #CALCULA A CORRENTE EM FUNÇÃO DO TEMPO
    I = ((m.exp(-0.125 * T) *m.cos(0.3307*T))+(1.1339*(m.exp(-0.125*T)*m.sin(0.3307*T))))
    V = ( ((5*m.exp(-0.125*T))*((2.6459*m.sin(0.3307*T))-(m.cos(0.3307*T))))-5 )
    #CALCULA A TENSÃO EM FUNÇÃO DO TEMPO
    #PREENCHE OS VETORES
    I_eixo.append(I)
    V_eixo.append(V)
    #TEMPO INSTANTÂNEO
    T_eixo.append(T)
    T += (Ti + step)

    #V_eixo.append(V)
    
plt.figure(1)
plt.subplot(2,1,1)
plt.plot(T_eixo, I_eixo, 'r',label='Corrente, I(t)')
plt.ylabel('$I(t)$', fontsize=12)
plt.xlabel('$t$', fontsize=12)
plt.grid(axis='both')
plt.legend(fontsize=10)

plt.subplot(2,1,2)
plt.plot(T_eixo, V_eixo, 'b',label='Tensão, I(t)')
plt.ylabel('$V(t)$', fontsize=12)
plt.xlabel('$t$', fontsize=12)
plt.grid(axis='both')
plt.legend(fontsize=10)
plt.show()