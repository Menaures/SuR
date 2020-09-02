import control as ctl
import matplotlib.pyplot as plt
import numpy as np


def system(K_P):
    num = [10]
    den = [10, 25, 11, 2]
    G = ctl.tf(num, den)
    G_0 = K_P * G
    return ctl.feedback(G_0)


def p_controller(K_kr):
    num = [10]
    den = [10, 25, 11, 2]
    G = ctl.tf(num, den)
    K_P = 0.5*K_kr
    G_0 = K_P * G
    return G_0 / (1 + G_0)


def pi_controller(K_kr, T_kr):
    num = [10]
    den = [10, 25, 11, 2]
    G = ctl.tf(num, den)
    K_P = 0.45*K_kr
    K_I = 0.53*K_kr/T_kr
    # K_PI = ctl.parallel(ctl.tf([K_I], [1, 0]), K_P)
    T_I = 0.85*T_kr
    K = ctl.tf([K_P*T_I, K_P], [T_I, 0])
    G_0 = K*G
    return ctl.feedback(G_0)


def pid_controller(K_kr, T_kr):
    num = [10]
    den = [10, 25, 11, 2]
    G = ctl.tf(num, den)
    K_P = 0.6*K_kr
    K_I = 1.2*K_kr/T_kr
    K_D = 0.072*K_kr*T_kr
    K_PID = ctl.parallel(ctl.tf([K_I], [1, 0]), ctl.tf([K_D, 0], [1]), K_P)
    G_0 = ctl.series(K_PID, G)
    return ctl.feedback(G_0)


def plot_step_response(system, file, save=False):
    plt.figure()
    plt.grid(True)
    plt.xlabel("$t$ in s")
    plt.ylabel("$y(t)$")
    filename = "plots/step_response_" + file

    t = np.linspace(0, 200, 400)
    # t = np.linspace(0, 8, 100)
    t, y = ctl.step_response(system, t)

    plt.plot(t, y)
    plt.legend([file])
    if save:
        plt.savefig(filename)
    else:
        plt.show()
    plt.clf()


if __name__ == '__main__':
    # sys1 = system(2.55)
    # plot_step_response(sys1, "1b_zoom", True)
    sys2 = p_controller(2.55)
    sys3 = pi_controller(2.55, 6)
    sys4 = pid_controller(2.55, 6)
    # plot_step_response(sys2, "P-Regler", True)
    # plot_step_response(sys3, "PI-Regler", True)
    # plot_step_response(sys4, "PID-Regler", True)
    # ctl.bode_plot([sys2, sys3, sys4], dB=True)
    # plt.show()
    gm, pm, wg, wp = ctl.margin(sys2)
    print("Gain margin: ", gm, "\nPhase margin: ", pm, "\nPhase crossover: "
                                                       "", wg,
          "\nGain crossover: ", wp)
    print("\n\n")
    gm, pm, wg, wp = ctl.margin(sys3)
    print("Gain margin: ", gm, "\nPhase margin: ", pm, "\nPhase crossover: "
                                                       "", wg,
          "\nGain crossover: ", wp)
    print("\n\n")
    gm, pm, wg, wp = ctl.margin(sys4)
    print("Gain margin: ", gm, "\nPhase margin: ", pm, "\nPhase crossover: "
                                                       "", wg,
          "\nGain crossover: ", wp)
    print("\n\n")
