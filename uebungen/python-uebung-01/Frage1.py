import numpy as np
import matplotlib.pyplot as plt
import toolbox_sr1

def f(x,u):
    """ Modellgleichung RLC-Kreis
    """

    # Parameter
    R = 25.0 # [Ohm]
    C = 50e-6 # [F]
    L = 200e-3 # [H]

    # ODE
    xdot = np.array([
        [x[1]/C],
        [1/L*(u - R*x[1] - x[0])]
    ])

    return np.squeeze(xdot)

def y(x,u):
    """ Ausgangsgleichung RLC-Kreis
    """
    return x[0]

# Simulationsparameter
delta_t = 0.0005 # Zeitschrit [s]
T_f = 0.5 # Zeitraum [s]
num_steps = int(T_f/delta_t) # Anzahl Simulationsschritte

# Steuerung
U = np.ones((num_steps,))

# Anfangsbedingung
x_0 = [0.0, 0.0]

# Nichtlineare Simulation
[X,Y,T] = toolbox_sr1.nlsim(f, x_0, U, delta_t, y)

# Plots
plt.figure(1)
plt.plot(T,Y)
plt.step(T[:-1],U, color = 'red')
plt.title('RLC Nichtlineare Simulation 1')
plt.xlabel('t [s]')
plt.ylabel('Spannung [V]')
plt.legend(['v_C(t)', 'v(t)'])

# Rechtecksignal
omega = 10*np.pi # [rad/s]
U = np.sign(np.sin(2*np.pi*omega*np.linspace(0, T_f, num_steps)))

# Nichtlineare Simulation
[X,Y,T] = toolbox_sr1.nlsim(f, x_0, U, delta_t, y)

# Plots
plt.figure(2)
plt.plot(T,Y)
plt.step(T[:-1],U, color = 'red')
plt.title('RLC Nichtlineare Simulation 2')
plt.xlabel('t [s]')
plt.ylabel('Spannung [V]')
plt.legend(['v_C(t)', 'v(t)'])
plt.show()