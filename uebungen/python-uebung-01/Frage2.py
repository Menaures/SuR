import numpy as np
import matplotlib.pyplot as plt
import control as ctrl

# Parameter
R = 25.0 # [Ohm]
C = 50e-6 # [F]
L = 200e-3 # [H]

# Systemmatrizen
A = np.array([[0.0, 1/C], [-1/L, -R/L]])
B = np.array([[0.0],[1/L]])
C = np.array([[1,0]])
D = np.array([[0.0]])

# System
sys = ctrl.ss(A,B,C,D)

# Simulationsparameter
delta_t = 0.0005 # Zeitschrit [s]
T_f = 0.5 # Zeitraum [s]
num_steps = int(T_f/delta_t) # Anzahl Simulationsschritte

# Steuerung
# U = np.ones((num_steps,))
omega = 10*np.pi # [rad/s]
U = np.sign(np.sin(2*np.pi*omega*np.linspace(0, T_f, num_steps)))
T = np.linspace(0,T_f, num_steps)

# Anfangsbedingung
x_0 = [0.0, 0.0]

# Simulate
[T,Y,X] = ctrl.forced_response(sys, T=T, U=U, X0=x_0)
#[T,Y] = ctrl.step_response(sys, T=T)

# Plots
plt.figure(1)
plt.plot(T,Y)
plt.step(T,U, color = 'red')
plt.title('RLC Nichtlineare Simulation 1')
plt.xlabel('t [s]')
plt.ylabel('Spannung [V]')
plt.legend(['v_C(t)', 'v(t)'])
plt.show()