"""
Copyright Jonas Bilal, Robin Vogt
Sources: Systemtheorie file toolbox_sr1
"""

import numpy as np
from matplotlib import pyplot as plt
import toolbox_sr1 as sr

def ode_f(x, u):
    """
    Implements the ode of an negative OpAmp with 
    a capacity.
    """
    R1 = 100  # 100 Ohm
    R2 = 900  # 900 Ohm
    C = 10 ** -3  # 3 mF
    xdot = np.array([-1 / (R2 * C) * x[0] - 1 / (R1 * C) * u])
    return xdot


def output_g(x, u):
    return x[0]


def simulation_constant_input():
    x0 = np.array([0])
    stepTime = 0.01
    numSteps = 500
    U = np.linspace(1, 1, numSteps)
    return sr.nlsim(ode_f, x0, U, stepTime, output_g)


def simulation_sin_input(w):
    f = w / (2 * np.pi) # Hz
    u = w*(np.linspace(0, 4*np.pi/w, 1000))
    U = np.sin(w*(np.linspace(0, 4 * np.pi / w, 1000)))
    x0 = np.array([0])
    stepTime = 4*np.pi/(w*1000)
    return [sr.nlsim(ode_f, x0, U, stepTime, output_g), u, U]


def main():
    plt.subplot(221)
    [X1, Y1, T1] = simulation_constant_input()
    plt.plot(T1, np.linspace(1, 1, 501), label = "Eingangsspannung")
    plt.plot(T1, Y1, label = "Ausgangsspannung", color = "green")
    plt.xlabel("$t$ in s")
    plt.ylabel("$v_C(t)$ in V")
    plt.title("Simulation der Ausgangsspannung $v_C(t)$ des invertierenden Verstärkers")
    plt.legend()
    plt.grid(True)

    plt.subplot(222)
    [X2, Y2, T2] = simulation_sin_input(0.1)[0]
    plt.plot(T2, np.sin(0.1 * T2), label = "Eingangssignal")
    plt.plot(T2, Y2, label = "Ausgangssignal", color = "red")
    plt.legend()
    plt.xlabel("$t$ in s")
    plt.ylabel("$v_C(t)$ in V")
    #plt.xlim(0,3 * 0.1 / (2*np.pi))
    plt.title("Simulation von Eingangs- und Ausgangsspannung bei $\omega = 0.1$ rad/s")
    plt.grid(True)

    plt.subplot(223)
    [X3, Y3, T3] = simulation_sin_input(10)[0]
    plt.plot(T3, Y3, label = "Ausgangssignal", color = "purple")
    plt.plot(T3, np.sin(10 * T3), label = "Eingangssignal")
    plt.legend()
    plt.xlabel("$t$ in s")
    plt.ylabel("$v_C(t)$ in V")
    plt.title("Simulation von Eingangs- und Ausgangsspannung bei $\omega = 10$ rad/s")
    plt.grid(True)

    plt.subplot(224)
    [X4, Y4, T4] = simulation_sin_input(100)[0]
    plt.plot(T4, Y4, label = "Ausgangssignal")
    plt.plot(T4, np.sin(100 * T4), label = "Eingangssignal")
    plt.xlabel("$t$ in s")
    plt.ylabel("$v_C(t)$ in V")
    plt.title("Simulation von Eingangs- und Ausgangsspannung bei $\omega = 100$ rad/s")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
   main() 
