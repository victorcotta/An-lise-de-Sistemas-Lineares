from scipy.integrate import solve_ivp
from scipy.integrate import quad
from scipy.misc import derivative
import matplotlib.pyplot as plt
import numpy as np
import math as m

#DERIVADA DA FUNÇÃO 1:
def der_F1(t, h_atual, A, alfa, Fi):
    if(h_atual < 0):
        h_atual = 0
    return ((Fi/A) - ((alfa/A)*h_atual))

#DERIVADA DA FUNÇÃO 2:
def der_F2(t, h_atual, A, beta, Fi):
    if(h_atual < 0):
        h_atual = 0
    return ((Fi/A) - ((beta/A)*m.sqrt(h_atual)))

#ERRO:
def erro(t, x_desejado, x_atual=None):
    if(x_atual is None):
        x_atual = x_desejado
        x_desejado = t
    return (x_desejado-x_atual)

#PARÂMETROS DO SISTEMA:
area = .2

##########################################################################################################################
##########################################################################################################################

#ITEM 1:

alfa = .4
beta = .4

#CONDIÇÕES INICIAIS:
h0 = 1
u = 0.04

#CASO 1:

#PARÂMETROS DE SIMULAÇÃO:
t = 0
tf = 3
step = .01

#LOOPING:
h_atual = h0
eixo_h = []
eixo_h.append(h_atual)
eixo_u = []
eixo_u.append(u)
eixo_t = []
eixo_t.append(t)

while(t < tf):
    #CONTROLE:
    eixo_u.append(u)

    #SOLUÇÃO DA 'EDO':
    sol = solve_ivp(der_F1, t_span=(t, t+step), y0=[h_atual], method='RK23', t_eval=[t, t+step], args=(area, alfa, u))

    h_atual = sol.y[0][-1] #+ np.random.normal(0, 0.002)
    eixo_h.append(h_atual)
    t += step
    eixo_t.append(t)

#PLOTAGEM DO GRÁFICO:
plt.figure(1)
plt.subplot(2,1,1)
#plt.ylim([0.74, 0.76])
plt.plot(eixo_t, eixo_h, 'k',label='Nível, h(t)')
plt.ylabel('$h(t)$', fontsize=12)
plt.xlabel('$t$', fontsize=12)
plt.grid(axis='both')
plt.legend(fontsize=10)
plt.subplot(2,1,2)
#plt.ylim([-0.02, 0.02])
plt.plot(eixo_t, eixo_u, 'b',label='Controle, u(t)')
plt.ylabel('$u(t)$', fontsize=12)
plt.xlabel('$t$', fontsize=12)
plt.grid(axis='both')
plt.legend(fontsize=10)
plt.show()

#CASO 2:

#CONDIÇÕES INICIAIS:
u = 0.04

#PARÂMETROS DE SIMULAÇÃO:
t = 0
tf = 3
step = .01

#LOOPING:
h_atual = h0
eixo_h.clear()
eixo_h.append(h_atual)
eixo_u.clear()
eixo_u.append(u)
eixo_t.clear()
eixo_t.append(t)

while(t < tf):
    #CONTROLE:
    eixo_u.append(u)

    #SOLUÇÃO DA 'EDO':
    sol = solve_ivp(der_F2, t_span=(t, t+step), y0=[h_atual], method='RK23', t_eval=[t, t+step], args=(area, beta, u))

    h_atual = sol.y[0][-1] #+ np.random.normal(0, 0.002)
    eixo_h.append(h_atual)
    t += step
    eixo_t.append(t)

#PLOTAGEM DO GRÁFICO:
plt.figure(2)
plt.subplot(2,1,1)
#plt.ylim([0.74, 0.76])
plt.plot(eixo_t, eixo_h, 'k',label='Nível, h(t)')
plt.ylabel('$h(t)$', fontsize=12)
plt.xlabel('$t$', fontsize=12)
plt.grid(axis='both')
plt.legend(fontsize=10)
plt.subplot(2,1,2)
#plt.ylim([-0.02, 0.02])
plt.plot(eixo_t, eixo_u, 'b',label='Controle, u(t)')
plt.ylabel('$u(t)$', fontsize=12)
plt.xlabel('$t$', fontsize=12)
plt.grid(axis='both')
plt.legend(fontsize=10)
plt.show()

##########################################################################################################################
##########################################################################################################################

#ITEM 2:

alfa = .4
beta = .3

#CONDIÇÕES INICIAIS:
h0 = 0.5
u = 0.04

#ALTURA DESEJADA (PONTO DE EQUILÍBRIO):
h_desejado = 0.6

#CASO 1:

#PARÂMETROS DE SIMULAÇÃO:
t = 0
tf = 3
step = .01

#LOOPING:
h_atual = h0
eixo_h.clear()
eixo_h.append(h_atual)
eixo_u.clear()
eixo_u.append(u)
eixo_t.clear()
eixo_t.append(t)

while(t < tf):
    #CONTROLE:
    if(h_atual < h_desejado):
        u = 1.2
    else:
        u = 0.2
    eixo_u.append(u)

    #SOLUÇÃO DA 'EDO':
    sol = solve_ivp(der_F1, t_span=(t, t+step), y0=[h_atual], method='RK23', t_eval=[t, t+step], args=(area, alfa, u))

    h_atual = sol.y[0][-1] #+ np.random.normal(0, 0.002)
    eixo_h.append(h_atual)
    t += step
    eixo_t.append(t)

#PLOTAGEM DO GRÁFICO:
plt.figure(3)
plt.subplot(2,1,1)
#plt.ylim([0.74, 0.76])
plt.plot(eixo_t, eixo_h, 'k',label='Nível, h(t)')
plt.ylabel('$h(t)$', fontsize=12)
plt.xlabel('$t$', fontsize=12)
plt.grid(axis='both')
plt.legend(fontsize=10)
plt.subplot(2,1,2)
#plt.ylim([-0.02, 0.02])
plt.plot(eixo_t, eixo_u, 'b',label='Controle, u(t)')
plt.ylabel('$u(t)$', fontsize=12)
plt.xlabel('$t$', fontsize=12)
plt.grid(axis='both')
plt.legend(fontsize=10)
plt.show()

#CASO 2:

#CONDIÇÕES INICIAIS:
u = 0.04

#PARÂMETROS DE SIMULAÇÃO:
t = 0
tf = 3
step = .01

#LOOPING:
h_atual = h0
eixo_h.clear()
eixo_h.append(h_atual)
eixo_u.clear()
eixo_u.append(u)
eixo_t.clear()
eixo_t.append(t)

while(t < tf):
    #CONTROLE:
    if(h_atual < h_desejado):
        u = 1.2
    else:
        u = 0.2
    eixo_u.append(u)

    #SOLUÇÃO DA 'EDO':
    sol = solve_ivp(der_F2, t_span=(t, t+step), y0=[h_atual], method='RK23', t_eval=[t, t+step], args=(area, beta, u))

    h_atual = sol.y[0][-1] #+ np.random.normal(0, 0.002)
    eixo_h.append(h_atual)
    t += step
    eixo_t.append(t)

#Plotagem do gráfico:
plt.figure(4)
plt.subplot(2,1,1)
#plt.ylim([0.74, 0.76])
plt.plot(eixo_t, eixo_h, 'k',label='Nível, h(t)')
plt.ylabel('$h(t)$', fontsize=12)
plt.xlabel('$t$', fontsize=12)
plt.grid(axis='both')
plt.legend(fontsize=10)
plt.subplot(2,1,2)
#plt.ylim([-0.02, 0.02])
plt.plot(eixo_t, eixo_u, 'b',label='Controle, u(t)')
plt.ylabel('$u(t)$', fontsize=12)
plt.xlabel('$t$', fontsize=12)
plt.grid(axis='both')
plt.legend(fontsize=10)
plt.show()

##########################################################################################################################
##########################################################################################################################

#ITEM 3:

alfa = .1
beta = .1

#CONDIÇÕES INICIAIS:
h0 = 0.5
u = 0.04

#ALTURA DESEJADA (PONTO DE EQUILÍBRIO):
h_desejado = 0.6

#CASO 1:

#PARÂMETROS DE SIMULAÇÃO:
t = 0
tf = 2
step = .01

#CONSTANTES DO CONTROLADOR 'PI':
Kp = 1
Ki = 1

#LOOPING:
h_atual = h0
eixo_h.clear()
eixo_h.append(h_atual)
eixo_u.clear()
eixo_u.append(u)
eixo_t.clear()
eixo_t.append(t)

while(t < tf):
    #CONTROLE:
    #PROPORCIONAL:
    prop = erro(h_desejado, h_atual)
    #INTEGRATIVO:
    integration, err = quad(erro, t, t+step, args=(h_desejado, h_atual))
    u = ((Kp*prop) + (Ki*integration))
    if(u < 0):
        u = 0
    eixo_u.append(u)

    #SOLUÇÃO DA 'EDO':
    sol = solve_ivp(der_F1, t_span=(t, t+step), y0=[h_atual], method='RK23', t_eval=[t, t+step], args=(area, alfa, u))

    h_atual = sol.y[0][-1] #+ np.random.normal(0, 0.002)
    eixo_h.append(h_atual)
    t += step
    eixo_t.append(t)

#PLOTAGEM DO GRÁFICO:
plt.figure(5)
plt.subplot(2,1,1)
#plt.ylim([0.74, 0.76])
plt.plot(eixo_t, eixo_h, 'k',label='Nível, h(t)')
plt.ylabel('$h(t)$', fontsize=12)
plt.xlabel('$t$', fontsize=12)
plt.grid(axis='both')
plt.legend(fontsize=10)
plt.subplot(2,1,2)
#plt.ylim([-0.02, 0.02])
plt.plot(eixo_t, eixo_u, 'b',label='Controle, u(t)')
plt.ylabel('$u(t)$', fontsize=12)
plt.xlabel('$t$', fontsize=12)
plt.grid(axis='both')
plt.legend(fontsize=10)
plt.show()

#CASO 2:

#PARÂMETROS DE SIMULAÇÃO:
t = 0
tf = 2
step = .01

#CONSTANTES DO CONTROLADOR 'PI':
Kp = 1
Ki = 1

#LOOPING:
h_atual = h0
eixo_h.clear()
eixo_h.append(h_atual)
eixo_u.clear()
eixo_u.append(u)
eixo_t.clear()
eixo_t.append(t)

while(t < tf):
    #CONTROLE:
    #PROPORCIONAL:
    prop = erro(h_desejado, h_atual)
    #INTEGRATIVO:
    integration, err = quad(erro, t, t+step, args=(h_desejado, h_atual))
    u = ((Kp*prop) + (Ki*integration))
    if(u < 0):
        u = 0
    eixo_u.append(u)

    #SOLUÇÃO DA 'EDO':
    sol = solve_ivp(der_F2, t_span=(t, t+step), y0=[h_atual], method='RK23', t_eval=[t, t+step], args=(area, beta, u))

    h_atual = sol.y[0][-1] #+ np.random.normal(0, 0.002)
    eixo_h.append(h_atual)
    t += step
    eixo_t.append(t)

#PLOTAGEM DO GRÁFICO:
plt.figure(6)
plt.subplot(2,1,1)
#plt.ylim([0.74, 0.76])
plt.plot(eixo_t, eixo_h, 'k',label='Nível, h(t)')
plt.ylabel('$h(t)$', fontsize=12)
plt.xlabel('$t$', fontsize=12)
plt.grid(axis='both')
plt.legend(fontsize=10)
plt.subplot(2,1,2)
#plt.ylim([-0.02, 0.02])
plt.plot(eixo_t, eixo_u, 'b',label='Controle, u(t)')
plt.ylabel('$u(t)$', fontsize=12)
plt.xlabel('$t$', fontsize=12)
plt.grid(axis='both')
plt.legend(fontsize=10)
plt.show()

##########################################################################################################################
##########################################################################################################################

#ITEM 5:

#ON-OFF:

alfa = .4
beta = .3

#CONDIÇÕES INICIAIS:
u = 0.04

#PARÂMETROS DE SIMULAÇÃO:
t = 0
tf = 150
step = .01

#LOOPING:
h_atual = h0
eixo_h.clear()
eixo_h.append(h_atual)
eixo_u.clear()
eixo_u.append(u)
eixo_t.clear()
eixo_t.append(t)

while(t < tf):
    if((t > (50-1e-6)) and (t < (50+1e-6))):
        h_atual = 0.4
    elif((t > (100-1e-6)) and (t < (100+1e-6))):
        h_atual = 0.8
    else:
        #CONTROLE:
        if(h_atual < h_desejado):
            u = 1.2
        else:
            u = 0.2

        #SOLUÇÃO DA 'EDO':
        sol = solve_ivp(der_F2, t_span=(t, t+step), y0=[h_atual], method='RK23', t_eval=[t, t+step], args=(area, beta, u))

        h_atual = sol.y[0][-1] #+ np.random.normal(0, 0.002)
    
    
    eixo_u.append(u)
    eixo_h.append(h_atual)
    t += step
    eixo_t.append(t)

#PLOTAGEM DO GRÁFICO:
plt.figure(7)
plt.subplot(2,1,1)
#plt.ylim([0.74, 0.76])
plt.plot(eixo_t, eixo_h, 'k',label='Nível, h(t)')
plt.ylabel('$h(t)$', fontsize=12)
plt.xlabel('$t$', fontsize=12)
plt.grid(axis='both')
plt.legend(fontsize=10)
plt.subplot(2,1,2)
#plt.ylim([-0.02, 0.02])
plt.plot(eixo_t, eixo_u, 'b',label='Controle, u(t)')
plt.ylabel('$u(t)$', fontsize=12)
plt.xlabel('$t$', fontsize=12)
plt.grid(axis='both')
plt.legend(fontsize=10)
plt.show()

#'PI':

alfa = .1
beta = .1

#CONDIÇÕES INICIAIS:
u = 0.04

#PARÂMETROS DE SIMULAÇÃO:
t = 0
tf = 150
step = .01

#CONSTANTES DO CONTROLADOR 'PI':
Kp = 50
Ki = 50

#LOOPING:
h_atual = h0
eixo_h.clear()
eixo_h.append(h_atual)
eixo_u.clear()
eixo_u.append(u)
eixo_t.clear()
eixo_t.append(t)

while(t < tf):
    if((t > (50-1e-6)) and (t < (50+1e-6))):
        h_atual = 0.4
    elif((t > (100-1e-6)) and (t < (100+1e-6))):
        h_atual = 0.8
    else:
        #CONTROLE:
        #PROPORCIONAL:
        prop = erro(h_desejado, h_atual)
        #Integrativo:
        integration, err = quad(erro, t, t+step, args=(h_desejado, h_atual))
        u = ((Kp*prop) + (Ki*integration))
        if(u < 0):
            u = 0

        #SOLUÇÃO DA 'EDO':
        sol = solve_ivp(der_F2, t_span=(t, t+step), y0=[h_atual], method='RK23', t_eval=[t, t+step], args=(area, beta, u))

        h_atual = sol.y[0][-1] #+ np.random.normal(0, 0.002)    

    eixo_u.append(u)
    eixo_h.append(h_atual)
    t += step
    eixo_t.append(t)

#PLOTAGEM DO GRÁFICO:
plt.figure(8)
plt.subplot(2,1,1)
#plt.ylim([0.74, 0.76])
plt.plot(eixo_t, eixo_h, 'k',label='Nível, h(t)')
plt.ylabel('$h(t)$', fontsize=12)
plt.xlabel('$t$', fontsize=12)
plt.grid(axis='both')
plt.legend(fontsize=10)
plt.subplot(2,1,2)
#plt.ylim([-0.02, 0.02])
plt.plot(eixo_t, eixo_u, 'b',label='Controle, u(t)')
plt.ylabel('$u(t)$', fontsize=12)
plt.xlabel('$t$', fontsize=12)
plt.grid(axis='both')
plt.legend(fontsize=10)
plt.show()