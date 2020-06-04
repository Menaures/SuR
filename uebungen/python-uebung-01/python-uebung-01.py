"""

"""

import sys
import SuR.uebungen.toolbox_sr1 as sr1
import numpy as np
import matplotlib.pyplot as plt

sys.path.append("D://Uni/SuR")


def f(x, u):
    # Parameter
    R = 25.0  # [Ohm]
    C = 50e-6  # [F]
    L = 200e-3  # [H]

    # ODE
    xdot = np.array((
        [x[1] / C],
        [1 / L * (u - R * x[1] - x[0])]
    ))

    return np.squeeze(xdot)


def y(x, u):
    return x[0]


# Simulationsparameter
T_f = 0.5  # [s]
delta_t = 0.0005  # [s]
num_steps = int(T_f / delta_t)

# Steuerung
U = np.ones((num_steps,))  # 1 V

# Anfangsbedingung
x_0 = [0.0, 0.0]

# Nichtlineare Simulation
[X, Y, T] = sr1.nlsim(f, x_0, U, delta_t, y)

# Plots
plt.figure(1)
plt.plot(T, Y)
plt.step(T[: -1], U, color='red')
plt.title('RLC Nichtlineare Simulation')
plt.xlabel('t [s]')
plt.ylabel('Spannung [V]')

# Rechtecksignal
omega = 10 * np.pi
U = np.sign(np.sin(2 * np.pi * omega * np.linspace(0, T_f, num_steps)))

# Nichtlineare Simulation
[X, Y, T] = sr1.nlsim(f, x_0, U, delta_t, y)

# Plots
plt.figure(2)
plt.plot(T, Y)
plt.step(T[: -1], U, color='red')
plt.title('RLC Nichtlineare Simulation')
plt.xlabel('t [s]')
plt.ylabel('Spannung [V]')
plt.show()
