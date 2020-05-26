import numpy as np
import matplotlib.pyplot as plt
import control as ctl

import toolbox_sr1 as sr1

#==========================================================================
# The function defines the ouptut mapping from state to output for the 
# tractor example
#   Input Parameter: state x, control input u; both are column vectors
#   Output Parameter: output y; is a column vector
#
# AUTHOR: Joerg Fischer, Lukas Klar
# Systems control and optimization laboratory, 
# Department of Microsystems Engineering (IMTEK),
# University of Freiburg
#
# REMARK: function is called from script "traktor_simulate"
#==========================================================================
# output is the second state (y-position of the tractor)

def tractor_out(x, u):
    xout = x[1]
    return xout

#==========================================================================
# The function defines the ODE of a tractor with the states X(the
# x-position), Y(the y-position) and beta(the orientation). The input is
# the steering angle alpha
# The nonlinear ODE is given by:
#   dX/dt = V * cos(beta)
#   dY/dt = V * sin(beta)
#   dbeta/dt = V/L * tan(alpha)
# Input Parameter: state x, control input u; both are column vectors 
# Output Parameter: dx/dt; is a column vector
#
# AUTHOR: Joerg Fischer, Lukas Klar
# Systems control and optimization laboratory, 
# Department of Microsystems Engineering (IMTEK),
# University of Freiburg
#==========================================================================

def tractor_ode_nl(x, u):
    # Parameter definition
    V = 5; # m/s
    L = 3; # m
    # ODE
    xdot = np.array([V * np.cos(x[2]), V * np.sin(x[2]), V / L * np.tan(u)],) #x[2]: beta bzw. Orientierungswinkel
    return xdot



# =========================================================================
# This file provides an example of a simulation of a nonlinear ordinary 
# differential equation (ODE). The example system ist the nonlinear 
# tractor model described in the lecture notes. 
# 
# AUTHOR: Joerg Fischer, Lukas Klar
# EDITED: Matthias Gramlich, Jochem De Schutter
# Systems control and optimization laboratory, 
# Department of Microsystems Engineering (IMTEK),
# University of Freiburg
# 
# File Created: 18.04.2016
# =========================================================================
#
# 
# Description of the tractor example:
# States of the tractor are the x-position (X), the y-position (Y) and the
# orientation angle (beta) according to the x-axis
#     x = [X(t); Y(t); beta(t)] 
# 
# Control input u of the system is the steering angle and the output is the
# distance from the x-axis aka y-position.
# 
# In the following, the tractor is simulated for 50 seconds. Starting with 
# the state [0; 5; 0] and a sinusoidal input.

###############  0.  Closing all open figures
 # close all

###############  1.  Definition of ODE dx/dt = f(x,u)
# This is implemented in the function: tractor_ode_nl()

###############  2.  Definition of output function y = g(x,u)
# This is implemented in the function: tractor_out()
  
###############  3.  Setting Simulation time parameters
# lenght of a single integration step in seconds
stepTime = 0.1
# number of time steps performed in simulation
numSteps = 500
# --> total simulated time is stepTime * numSteps = 50s

###############  4.  Definition of initial state
x_0 = [0, 5, 0]

###############  5.  Definition of control input trajectory
# U is a sequence of control inputs u, where u is a vector if the 
# system has more than one inpput. The column index of U represents the 
# time index when a particular u (rows of U) is applied to the system.

# Create a sinusoidal input sequence (2 periods with amplitude 0.05)
U = 0.05*np.sin(np.linspace(0, 4*np.pi, numSteps))

###############  6.  Simulation of system
# Input Parameter: see steps 1 to 4 above and documentation of nlsim
# Output Parameter: X, Y are the state and output trajectory, 
#                  T is the timegrid belonging to X and Y

[X, Y, T] = sr1.nlsim(tractor_ode_nl, x_0, U, stepTime, tractor_out)

###############  7.  Plotting output and Legend and labels

plt.figure()

plt.plot(T,  Y, 'g:', label='$Y_{nonlin}$')
plt.legend(loc='upper center', shadow=True, fontsize='x-large')
plt.title('Ausgangssignal des gesteuerten Traktors')
plt.xlabel('Zeit in s')
plt.ylabel('Ort in m')

plt.show()











    
