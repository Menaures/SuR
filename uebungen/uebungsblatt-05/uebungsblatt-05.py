"""
Simulation of PT2 elements
"""

import sys
import SuR.uebungen.toolbox_sr1 as sr1
import numpy as np
import scipy.integrate as integrate
import matplotlib.pyplot as plt
import control as ctl

sys.path.append("..")


def ode_f1(x, u):
    """

    """
    A = np.array([[0, 1],
                  [-1, -2]])
    B = np.array([0, 1]).T

    xdot = np.dot(A, x) + np.dot(B, u)
    return xdot


def ode_f2(x, u):
    """

    """
    A = np.array([[0, 1],
                  [-1, -1]])
    B = np.array([0, 1]).T

    xdot = np.dot(A, x) + np.dot(B, u)
    return xdot


def ode_f3(x, u):
    """

    """
    A = np.array([[0, 1],
                  [-1, -4]])
    B = np.array([0, 1]).T

    xdot = np.dot(A, x) + np.dot(B, u)
    return xdot


def ode_f4(x, u):
    """

    """
    A = np.array([[0, 1],
                  [-1 / 4, -1 / 2]])
    B = np.array([0, 1]).T

    xdot = np.dot(A, x) + np.dot(B, u)
    return xdot


def output_g(ODE, u):
    """
    Return the output of the given ODE.

    Args:
        ODE: an array consisting of the states of system.
    Returns:
        return the first element of the state which is the output function.
    """
    return ODE[0]


def control_sim():
    A = np.array([[-4, 3, -3],
                  [-1, -1, -2],
                  [-2, 4, -4]])
    B = np.array([[1],
                  [0],
                  [2]])
    C = np.array([3, 1, 0])
    D = np.array([0])

    return ctl.ss(A, B, C, D)


def main():
    # plot exercise 1 e)

    # simulation parameter
    step_time = 0.1  # in sec
    sim_time = 20  # in sec
    num_steps = int(sim_time / step_time)
    x_0 = np.array([0, 0])
    U = np.array(np.linspace(1, 1, num_steps))

    # plot
    plt.figure()
    ODE = ode_f1
    [X1, Y1, T1] = sr1.nlsim(ODE, x_0, U, step_time, output_g)
    plt.plot(T1, Y1, label="Funktion 1", color="green")
    plt.xlabel("$t$ in s")
    plt.ylabel("$y(t)")
    plt.legend()
    plt.grid(True)

    ODE = ode_f2
    [X2, Y2, T2] = sr1.nlsim(ODE, x_0, U, step_time, output_g)
    plt.plot(T2, Y2, label="Funktion 2", color="blue")
    plt.xlabel("$t$ in s")
    plt.ylabel("$y(t)")
    plt.legend()
    plt.grid(True)

    ODE = ode_f3
    [X3, Y3, T3] = sr1.nlsim(ODE, x_0, U, step_time, output_g)
    plt.plot(T3, Y3, label="Funktion 3", color="red")
    plt.xlabel("$t$ in s")
    plt.ylabel("$y(t)")
    plt.legend()
    plt.grid(True)

    ODE = ode_f4
    [X4, Y4, T4] = sr1.nlsim(ODE, x_0, U, step_time, output_g)
    plt.plot(T4, Y4, label="Funktion 4", color="yellow")
    plt.xlabel("$t$ in s")
    plt.ylabel("$y(t)")
    plt.legend()
    plt.grid(True)

    plt.savefig("plots/plots_1e.png")
    plt.clf()

    # plot exercise 1 g)
    plt.figure()
    plt.plot(T4, Y4, label="Funktion 4", color="yellow")
    plt.xlabel("$t$ in s")
    plt.ylabel("$y(t)")
    plt.legend()
    plt.grid(True)

    # plot
    ODE = ode_f2
    [X5, Y5, T5] = sr1.nlsim(ODE, x_0, U, step_time, output_g)
    T5 = np.linspace(1, 20, int(num_steps / 2))
    plt.plot(T5, Y5[:int((len(Y5) - 1) / 2)], label="Funktion 2 verlangsamt",
             color="blue")
    plt.xlabel("$t$ in s")
    plt.ylabel("$y(t)$")
    plt.legend()
    plt.grid(True)

    plt.savefig("plots/plots_1g.png")
    plt.clf()

    # control simulation
    sys1 = control_sim()

    # step response
    [T6, Y6] = ctl.step_response(sys1)
    plt.figure()
    plt.plot(T6, Y6, label="Sprungantwort")
    plt.xlabel("$t$ in s")
    plt.ylabel("$y(t)$")
    plt.legend()
    plt.grid(True)
    plt.savefig("plots/plot_2b.png")
    plt.clf()

    # calculate t_settling
    dc_gain = Y6[-1]
    for i in range(len(Y6)):
        if Y6[i] <= dc_gain + abs(0.02 * dc_gain):
            print(T6[i])
            break

    # initial response
    X_0 = [1, -1, 1]
    [T7, Y7] = ctl.initial_response(sys1, X0=X_0)
    plt.figure()
    plt.plot(T7, Y7, label="Antwortfunktion")
    plt.xlabel("$t$ in s")
    plt.ylabel("$y(t)$")
    plt.legend()
    plt.grid(True)
    plt.savefig("plots/plot_2c.png")

    # forced response
    # Rectangular wave signal
    sim_time = 5
    num_steps = 2000
    omega1 = 0.2 * np.pi
    U = np.multiply(np.sign(np.sin(2 * np.pi * omega1 * np.linspace(0,
                                                                     sim_time,
                                                                     num_steps))),
                     0.5)
    T8 = np.linspace(0, sim_time, num_steps)
    [T8, Y8, X8] = ctl.forced_response(sys1, T=T8, U=U)
    plt.figure()
    plt.plot(T8, Y8, label="Ausgang $y(t)$")
    plt.xlabel("$t$ in s")
    plt.ylabel("$y(t)$")
    plt.plot(T8, U, label="Eingang $u(t)$")
    plt.legend()
    plt.grid(True)
    plt.savefig("plots/plot_2d_0.1.png")
    plt.clf()

    # forced response
    # Rectangular wave signal
    sim_time = 0.5
    num_steps = 2000
    omega1 = 2 * np.pi
    U = np.multiply(np.sign(np.sin(2 * np.pi * omega1 * np.linspace(0,
                                                                     sim_time,
                                                                     num_steps))),
                     0.5)
    T9 = np.linspace(0, sim_time, num_steps)
    [T9, Y9, X9] = ctl.forced_response(sys1, T=T9, U=U)
    plt.figure()
    plt.plot(T9, Y9, label="Ausgang $y(t)$")
    plt.xlabel("$t$ in s")
    plt.ylabel("$y(t)$")
    plt.plot(T9, U, label="Eingang $u(t)$")
    plt.legend()
    plt.grid(True)
    plt.savefig("plots/plot_2d_1.png")
    plt.clf()

    # forced response
    # Rectangular wave signal
    sim_time = 0.05
    num_steps = 2000
    omega1 = 20 * np.pi
    U = np.multiply(np.sign(np.sin(2 * np.pi * omega1 * np.linspace(0,
                                                                     sim_time,
                                                                     num_steps))),
                     0.5)
    T10 = np.linspace(0, sim_time, num_steps)
    [T10, Y10, X10] = ctl.forced_response(sys1, T=T10, U=U)
    plt.figure()
    plt.plot(T10, Y10, label="Ausgang $y(t)$")
    plt.xlabel("$t$ in s")
    plt.ylabel("$y(t)$")
    plt.plot(T10, U, label="Eingang $u(t)$")
    plt.legend()
    plt.grid(True)
    plt.savefig("plots/plot_2d_10.png")
    plt.clf()


if __name__ == "__main__":
    main()
