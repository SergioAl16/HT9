# -*- coding: utf-8 -*-
"""
Universidad del Valle de Guatemala
Métodos Numéricos
Sección 40
Sergio Alejandro Vasquez Marroquin - 161259
31/10/2023

METODO DE SOLUCIONES NUMERICAS DE EDO - HT9

"""

import numpy as np
import matplotlib.pyplot as plt

def euler1(f,h,x0,y0,xf):
    
    xlista = [x0]
    ylista = [y0]
    
    x = x0
    y = y0
    
    while x < xf:
        
        y = y + eval(f)*h
        x = x + h
        
        xlista.append(x)
        ylista.append(y)

    """plt.figure(figsize = (10,8))
    plt.plot(xlista, ylista, 'b')
    plt.title('EDO')
    plt.xlabel('r')
    plt.ylabel('y')"""
        
    return xlista,ylista

def euler2(f,g,h,t0,x0,y0,tf):
    
    tlista = [t0]
    xlista = [x0]
    ylista = [y0]
    
    t = t0
    x = x0
    y = y0
    
    while t < tf:
        
        x = x + eval(f)*h
        y = y + eval(g)*h
        t = t + h
        
        tlista.append(t)
        xlista.append(x)
        ylista.append(y)

    """plt.figure(figsize = (10,8))
    plt.plot(tlista, xlista, 'g')
    plt.plot(tlista, ylista, 'b')
    plt.title('EDO')
    plt.xlabel('t')
    plt.ylabel('x & y')"""
        
    return xlista,ylista,tlista

def puntoMedio1(f,h,x0,y0,xf):
    
    xlista = [x0]
    ylista = [y0]
    
    x = x0
    y = y0
    
    while x < xf:
        
        yant = y
        
        fxy = eval(f)
        
        y = y + fxy*h/2 #este es y_1/2
        x = x + h/2 #este es el x_1/2
        
        y = yant + eval(f)*h
        x = x + h/2
        
        xlista.append(x)
        ylista.append(y)

    return xlista,ylista

def heun1(f,h,x0,y0,xf):
    
    xlista = [x0]
    ylista = [y0]
    
    x = x0
    y = y0
    
    while x < xf:
        
        yant = y
        
        fxy = eval(f)
        
        y = y + fxy*h 
        x = x + h
        
        fxy_0 = eval(f)
        
        y = yant + (fxy + fxy_0)/2 * h
        
        xlista.append(x)
        ylista.append(y)
    
    return xlista,ylista

def RK41(f,h,x0,y0,xf):
    
    xlista = [x0]
    ylista = [y0]
    
    x = x0
    y = y0
    
    while x < xf:
        yactual = y
        
        k1 = eval(f)
        
        x = x + h/2
        y = yactual + k1*h/2
        k2 = eval(f)
        
        y = yactual + k2*h/2
        k3 = eval(f)
        
        x = x + h/2
        y = yactual + k3*h
        k4 = eval(f)
        
        y = yactual + (k1+2*k2+2*k3+k4)*h/6
        
        xlista.append(x)
        ylista.append(y)

    return xlista,ylista

def Ralston1(f,h,x0,y0,xf):
    
    xlista = [x0]
    ylista = [y0]
    
    x = x0
    y = y0
    
    while x < xf:
        yactual = y
        
        k1 = eval(f)
        
        x = x + 3/4*h
        y = yactual + (3/4)*k1*h
        k2 = eval(f)
        
        y = yactual + ((1/3)*k1+(2/3)*k2)*h
        
        xlista.append(x)
        ylista.append(y)

    return xlista,ylista

def RK31(f,h,x0,y0,xf):
    
    xlista = [x0]
    ylista = [y0]
    
    x = x0
    y = y0
    
    while x < xf:
        yactual = y
        
        k1 = eval(f)
        
        x = x + h/2
        y = yactual + (1/2)*k1*h
        k2 = eval(f)
        
        x = x + h/2
        y = yactual - k1*h + 2*k2*h
        k3 = eval(f)
        
        y = yactual + (k1+4*k2+k3)*h/6
        
        xlista.append(x)
        ylista.append(y)

    return xlista,ylista

# EJERCICIO EN CLASE 1
"""euler1("(2*(y**2)-(x**2))/(x*y)",0.01,1,2,10)

x_new = np.linspace(1, 10, 1000)
y_new = np.sqrt((x_new**2)+3*(x_new**4))
plt.plot(x_new,y_new, 'r')
plt.show()"""

# EJERCICIO EN CLASE 2
#euler2("5*x-8*y+np.exp(t)","y-4*x+t**2",0.1,0,0,0,5)

# EJERCICIO EN CLASE 3
#euler2("y","-9.8*np.sin(x)",0.1,0,3,0,10)
f = "(x-(x+1)*y)/x"
h = 0.25
x0 = np.log(2)
xf = 10
y0 = 1

RK4 = RK41(f,h,x0,y0,xf)
puntoMedio = puntoMedio1(f,h,x0,y0,xf)
heun = heun1(f,h,x0,y0,xf)
euler = euler1(f,h,x0,y0,xf)
ralston = Ralston1(f,h,x0,y0,xf)
RK3 = RK31(f,h,x0,y0,xf)

f_new = "1-(1/x)+(2*np.exp(-x))/x"
x_new = np.linspace(np.log(2), 10, 1000)
x = x_new
y_new = eval(f_new)

plt.figure(figsize = (10,8))
plt.plot(x_new, y_new, 'r', label = 'Real')
plt.plot(puntoMedio[0], puntoMedio[1], 'y', label = 'PuntoMedio')
plt.plot(RK4[0], RK4[1],'c', label = 'Runge-Kuta')
plt.plot(heun[0], heun[1],'b', label = 'Heun')
plt.plot(euler[0], euler[1],'g', label = 'Euler')
plt.plot(ralston[0], ralston[1],'m', label = 'Ralston')
plt.plot(RK3[0], RK3[1],'black', label = 'RK3')
plt.title('EDO')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()