import sys

import SuR.uebungen.toolbox_sr1 as sr1
import control as ctl
import numpy as np
import matplotlib.pyplot as plt

sys.path.append("..")


def system_1():
    num = [1, 2]
    den = [1, 1, 5, 2]
    return ctl.tf(num, den)


def exercise_1():
    # system transfer function
    num = [1, 2]
    den = [1, 1, 5, 2]

    # create system
    sys_1 = system_1()

    # nyquist plot
    plt.figure()
    ctl.nyquist_plot([sys_1])

    # plot circle
    t = np.linspace(0, 2*np.pi, 100)
    plt.plot(np.cos(t), np.sin(t))

    # save plot
    plt.savefig("plots/nyquist_plot.png")
    plt.clf()
    plt.figure()

    # bode plot
    ctl.bode_plot([sys_1], dB=True)

    # save plot
    plt.savefig("plots/bode_plot.png")
    plt.clf()


def system_2(k_s, T, T_t):
    num = [k_s]
    den = [T, 1]
    sys = ctl.tf(num, den)
    sys = sr1.delay(sys, T_t)
    return sys


def exercise_2():
    # get unknown step response
    t, y = sr1.unknown_step()

    # plot unknown step response
    plt.figure()
    plt.grid(True)
    plt.xlabel("$t$ in s")
    plt.ylabel("$y(t)$")
    plt.plot(t, y)

    # create system
    k_s = 1.38
    T = 1.1
    T_t = 2.2
    sys_2 = system_2(k_s, T, T_t)

    # plot system step response
    t = np.linspace(0, 50, 100)
    t, y = ctl.step_response(sys_2, t)
    plt.plot(t, y)

    # plot PID controlled system step response
    K_P = 1.2*T/(k_s*T_t)
    K_I = 0.6*T/(k_s*T_t**2)
    K_D = 0.6*T/k_s
    t, y = sr1.unknown_step(K_P, K_I, K_D)
    plt.plot(t, y)

    # save plot
    plt.savefig("plots/step_response.png")
    plt.clf()


if __name__ == '__main__':
    exercise_1()
    exercise_2()
