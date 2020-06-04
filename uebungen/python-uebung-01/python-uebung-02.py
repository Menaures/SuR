"""

"""

import sys
import SuR.uebungen.toolbox_sr1 as sr1
import numpy as np
import matplotlib.pyplot as plt

sys.path.append("D://Uni/SuR")


# Parameter
R = 25.0  # [Ohm]
C = 50e-6  # [F]
L = 200e-3  # [H]

# Systemmatrizen (xdot = Ax + Bu, y = Cx + Du)
A = np.array([[0.0, 1/C], [-1/L, -R/L]])
B = np.array([[0.0], [1/L]])
C = np.array([1.0, 0.0])
D = np.array([[0.0]])

# Definiere System
sys = ctl.ss(A, B, C, D)

# Simulationsparameter
T_f = 0.5  # [s]
delta_t = 0.0005  # [s]
num_steps = int(T_f/delta_t)
T = np.linspace(0, T_f, num_steps)

# Steuerung
U = np.ones((num_steps, ))  # 1 V

# Anfangsbedingung
x_0 = [0.0, 0.0]
[T, Y] = ctl.step_response(sys, T=T)


# Plots
plt.figure(1)
plt.plot(T, Y, marker='o')
plt.step(T, U, color='red')
plt.title('RLC Nichtlineare Simulation')
plt.xlabel('t [s]')
plt.ylabel('Spannung [V]')
plt.legend(['v_c(t)', 'v(t)'])
plt.show()