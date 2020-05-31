"""
Soruches: tractor.py from Joerg Fischer, Lukas Klar
"""

import numpy as np
import matplotlib.pyplot as plt
import control as ctl

import toolbox_sr1 as sr1

def sink_out(x, u):
    xout = x[0]
    return xout

def sink_ode_nl(x, u):
    # Parameter definition
    k1 = 0.6
    k2 = 0.1
    uss = 2.4
    Th = 340
    Ta = 300
    # ODE
    xdot = np.array([-k1 * np.sqrt(x[0]) + u, -k2 * (x[1] - Ta) + (Th-x[1])/x[0] * u],) 
    return xdot

def sink_ode_linear(deltax, deltau):
    k1 = 0.6
    k2 = 0.1
    uss = 2.4
    Th = 340
    Ta = 300
    # ODE
    A = np.array([[-0.075, 0], [-0.15, -0.25]])
    B = np.array([deltau, deltau]) 
    return  np.matmul(A, deltax) + B

stepTime = 0.1
# number of time steps performed in simulation
numSteps = 500
# --> total simulated time is stepTime * numSteps = 50s

###############  4.  Definition of initial state
x_0 = [1,300]

###############  5.  Definition of control input trajectory
# U is a sequence of control inputs u, where u is a vector if the 
# system has more than one inpput. The column index of U represents the 
# time index when a particular u (rows of U) is applied to the system.

U = np.array(np.linspace(2.4, 2.4, numSteps))

###############  6.  Simulation of system
# Input Parameter: see steps 1 to 4 above and documentation of nlsim
# Output Parameter: X, Y are the state and output trajectory, 
#                  T is the timegrid belonging to X and Y

[X1, Y1, T1] = sr1.nlsim(sink_ode_linear, x_0, U, stepTime, sink_out)
[X, Y, T] = sr1.nlsim(sink_ode_nl, x_0, U, stepTime, sink_out) 

###############  7.  Plotting output and Legend and labels

fig, ax1 = plt.subplots()
ax1.plot(T, X[:,0], 'b--',  label = "Zustand $x_1$")
ax1.set_ylabel("Mass $m$ in kg")
ax1.legend(loc='upper center', shadow=True, fontsize='x-large')
ax2 = ax1.twinx()
ax2.plot(T, X[:,1], 'g--',  label = "Zustand $x_2$")
plt.title('Zustände des Systems')
plt.xlabel("Zeit $t$ in Sekunden")
plt.ylabel("Temperatur $T$ in K", color = "green")
ax2.tick_params(axis='y', labelcolor= "green")
plt.show()
