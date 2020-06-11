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
                 [-1/4, -1/2]])
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


def control_sim(x, u):
    A = np.array([[-4, 3, -3],
                  [-1, -1, -2],
                  [-2, 4, -4]])
    B = np.array([1, 0, 2]).T
    C = np.array([3, 1, 0])
    D = np.array([0])

    ctl.ss(A, B, C, D)


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
    plt.title("Verzögerungsglieder zweiter Ordnung")
    plt.legend()
    plt.grid(True)

    ODE = ode_f2
    [X2, Y2, T2] = sr1.nlsim(ODE, x_0, U, step_time, output_g)
    plt.plot(T2, Y2, label="Funktion 2", color="blue")
    plt.xlabel("$t$ in s")
    plt.ylabel("$y(t)")
    plt.title("Verzögerungsglieder zweiter Ordnung")
    plt.legend()
    plt.grid(True)

    ODE = ode_f3
    [X3, Y3, T3] = sr1.nlsim(ODE, x_0, U, step_time, output_g)
    plt.plot(T3, Y3, label="Funktion 3", color="red")
    plt.xlabel("$t$ in s")
    plt.ylabel("$y(t)")
    plt.title("Verzögerungsglieder zweiter Ordnung")
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

    # simulation parameter for slow motion
    step_time = 0.1  # in sec
    sim_time = 20  # in sec
    num_steps = int(sim_time / step_time)
    x_0 = np.array([0, 0])
    U = np.array(np.linspace(1, 1, num_steps))

    # plot
    ODE = ode_f2
    [X5, Y5, T5] = sr1.nlsim(ODE, x_0, U, step_time, output_g)
    T5 = np.linspace(1, 20, int(num_steps/2))
    plt.plot(T5, Y5[:int((len(Y5)-1)/2)], label="Funktion 2 verlangsamt",
             color="blue")
    plt.xlabel("$t$ in s")
    plt.ylabel("$y(t)$")
    plt.legend()
    plt.grid(True)

    # plt.show()
    plt.savefig("plots/plots_1g.png")
    plt.clf()


if __name__ == "__main__":
    main()
