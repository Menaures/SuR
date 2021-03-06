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

sys.path.append("..")


def f(x, u):
    # parameters
    R_1 = 100  # [Ohm]
    R_2 = 900  # [Ohm]
    C = 1e-3  # [F]

    # ODE
    xdot = -1/(R_2*C)*x -1/(R_1*C)*u

    return xdot


def y(x, u):
    return x


# simulation parameters
T_f = 5  # [s]
delta_t = 0.0005  # [s]
num_steps = int(T_f / delta_t)

# input control
U = np.linspace(1, 1, num_steps)

# initial conditions
x_0 = 0

# linear simulation
[X, Y, T] = sr1.nlsim(f, x_0, U, delta_t, y)

# plot
plt.figure(1)
plt.plot(T, Y)
plt.step(T[: -1], U, color='red')
plt.title('Integrator non-linear simulation')
plt.xlabel('t [s]')
plt.ylabel('Voltage [V]')
plt.legend(["$v_A(t)$", "$v_E(t)$"])
plt.savefig("plot.png")
