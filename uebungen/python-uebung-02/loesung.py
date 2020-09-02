import numpy as np
import matplotlib.pyplot as plt
import toolbox_sr1
import control


def f_fwd(x,u):
    """ Modellgleichung Traktor
    """
    
    # Parameter
    V  = 5 # [m/s]
    L1 = 6 # [m]
    L2 = 4 # [m]

    # ODE
    xdot = np.array([
        [V/L1*np.tan(u)],
        [V/L2*np.cos(x[0] - x[1] - np.pi/2)]
    ])

    return np.squeeze(xdot)

def f_bwd(x,u):
    """ Modellgleichung Traktor
    """
    
    # Parameter
    V  = -5 # [m/s]
    L1 = 6 # [m]
    L2 = 4 # [m]

    # ODE
    xdot = np.array([
        [V/L1*np.tan(u)],
        [V/L2*np.cos(x[0] - x[1] - np.pi/2)]
    ])

    return np.squeeze(xdot)

def g(x,u):
    """ Ausgangsgleichung Traktor
    """
    return x[1]

# Parameter
V = 5.0 # [m/s]
L1 = 6.0 # [m]
L2 = 4.0 # [m]

# Zustandsraummodell
A = np.matrix([
    [0, 0],
    [V/L2, -V/L2]
])

B = np.matrix([
    [V/L1],
    [0]
])

C = np.matrix([
    [0,1]
])

D = np.matrix([[0]])

# Zustandsraummodell erstellen mit "ss"
ss_fwd = control.ss(A,B,C,D)

# Pole berechnen
pole_fwd = control.pole(ss_fwd)
print(pole_fwd)

# Eigenvektoren berechnen --> Grenzstabil
U, V = np.linalg.eig(A)
print(U,V)

# Rückwärtsfahren
V = -5 # [m/s]
A = np.matrix([
    [0, 0],
    [V/L2, -V/L2]
])
B = np.matrix([
    [V/L1],
    [0]
])
# Zustandsraummodell erstellen mit "ss"
ss_bwd = control.ss(A,B,C,D)

# Pole berechnen
pole_bwd = control.pole(ss_bwd)
print(pole_bwd)

# Simulation der linearen Systeme
num_steps = int(10.0/0.01)
T = np.linspace(0, 10.0, num_steps)
U = 0.3*np.ones((num_steps,))
# U[100:] = 0.0
[T1, Y_fwd, X_fwd] = control.forced_response(ss_fwd, T = T, U = U, X0 = [0,0])
[T1, Y_bwd, X_bwd] = control.forced_response(ss_bwd, T = T, U = U, X0 = [0,0])

[X_fwd_nl, Y_fwd_nl, T_nl] = toolbox_sr1.nlsim(f_fwd, [0.0, 0.0], U, 0.01, g)
[X_bwd_nl, Y_bwd_nl, T_nl] = toolbox_sr1.nlsim(f_bwd, [0.0, 0.0], U, 0.01, g)

# Plotten der Simulation
plt.figure()
plt.plot(T1, X_fwd[0,:], color='blue')
plt.plot(T1, X_fwd[1,:], color='red')
plt.plot(T_nl, X_fwd_nl[:,0],color ='blue',linestyle='--')
plt.plot(T_nl, X_fwd_nl[:,1],color='red',linestyle='--')
plt.legend(['beta','gamma', 'beta_nl', 'gamma_nl'])
plt.grid(True)
plt.show()

plt.figure()
plt.plot(T1, X_bwd[0,:], color='blue')
plt.plot(T1, X_bwd[1,:], color='red')
plt.plot(T_nl, X_bwd_nl[:,0],color ='blue',linestyle='--')
plt.plot(T_nl, X_bwd_nl[:,1],color ='red',linestyle='--')
plt.legend(['beta','gamma','beta_nl','gamma_nl'])
plt.ylim([-4, 4])
plt.grid(True)
plt.show()