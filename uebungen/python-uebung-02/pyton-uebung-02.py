import sys
import control as ctl
import numpy as np
import matplotlib.pyplot as plt

sys.path.append("D://Uni/SuR")


def sys_lin(beta, gamma):
    """A linearized model of a tractor with trailer."""

    # parameters
    L_1 = 6  # in m
    L_2 = 4  # in m
    V = 5  # in m/s


    A = np.array([[0, 0],
                  -V / L_2 * np.sin(beta - gamma - np.pi / 2),
                  -V / L_2 * np.sin(beta - gamma - np.pi / 2)])
