"""
Simulation of a PT1-Element in form of an inverting op-amp circuit with
negative feedback over a capacitor.

u(t) = v_E(t):          input voltage
x(t) = v_C(t):          capacitor voltage
y(t) = x(t) = v_A(t):   output voltage

"""

import sys
import SuR.uebungen.toolbox_sr1 as sr1
import numpy as np
import scipy.integrate as integrate
import matplotlib.pyplot as plt

sys.path.append("D://Uni/SuR")


def f(x, u):
    # parameters
    R_1 = 100  # [Ohm]
    R_2 = 900  # [Ohm]
    C = 1e-3  # [F]

    # ODE
    xdot = -1/(R_2*C)*x[0] -1/(R_1*C)*u

    return xdot


def y(x, u):
    # parameters
    R_1 = 100  # [Ohm]
    R_2 = 900  # [Ohm]
    C = 1e-3  # [F]

    # ODE
    y = x_0 - integrate.quad((u/R_1 + x[0]/R_2), 0, delta_t)[0]/C

    return y


# simulation parameters
T_f = 5  # [s]
delta_t = 0.0005  # [s]
num_steps = int(T_f / delta_t)

# input control
U = [1]

# initial conditions
x_0 = 0

# linear simulation
[X, Y, T] = sr1.nlsim(f, x_0, U, delta_t, y)

# plot
plt.figure(1)
plt.plot(T, Y)
plt.step(T[: -1], U, color='red')
plt.title('Integrator linear simulation')
plt.xlabel('t [s]')
plt.ylabel('Spannung [V]')