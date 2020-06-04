"""
This module provides a Python implementation of functions presented in the
lecture "Systemtheorie and Regelungstechnik" at the Systems Control and 
Optimization Laboratory of the Department of Microsystems Engineering (IMTEK) 
of the University of Freiburg. 

@authors: Jochem De Schutter, Matthias Gramlich, Jörg Fischer

File Created: 21.05.2020
"""

# imports
import numpy as np

# Runge-Kutta integrator function
"""
Function  rk() 
   implements the Runge-Kutta algorithm presented in the lecture notes at the  
  end of chapter 2.8. Given the inital state x and control input u of a 
   system, which is represented by an ODE of the form dx/dt = f(x,u ), the 
   algorithm calculates the value of the state after the given time delta_t. 
   The control input is assumed to be constant during this time step.

Synopsis:
   x = rk(my_ode, delta_t, x, u)

Input Parameter:
   - my_ode: ODE of the form dx/dt = f(x,u) that has to be integrated
              <function handle to function f where f is implemented such that 
              inputs x, u and outputs dx/dt are scalars or ROW vectors of type 
              numpy array>
   - x: initial state at beginning of time interval
         <ROW vector of type numpy array> or scalar
   - delta_t: length of integration interval
               <positive float>
   - u: control input of ode; assumed to be constant during time interval;
         <ROW vector of type numpy array> or scalar

Output Parameter:
   - state at the end of time interval
      <ROW vector of type numpy array> or scalar
"""


def rk(my_ode, delta_t, x, u):
    k1 = my_ode(x, u)
    k2 = my_ode(x + delta_t * k1 / 2., u)
    k3 = my_ode(x + delta_t * k2 / 2., u)
    k4 = my_ode(x + delta_t * k3, u)

    return (x + delta_t * (k1 + 2 * k2 + 2 * k3 + k4) / 6.)


# Nonlinear simulator
"""
Function: nlsim()
   performs a forward simulation of the system that is described by the given
   ODE dx/dt = f(x,u) and given output function y=g(x, u). The simulation 
   starts at t=0 at the given initial state x_0. Then, one simulation step of 
   length delta_t is performed for each control input in the provided 
   control input trajectory U. It is assumed that a control input is constant 
   during the time interval it is applied. 
  
Synopsis:
   X, Y, T = nlsim(my_ode, x_0, U, delta_t, my_outFunc)
  
Input Parameter:
   - my_ode: ODE of the form dx(t)/dt = f(x(t),u(t)) that has to be integrated
         <function handle to function f. The inputs and outputs of function f 
         must either be scalars or ROW vectors of type numpy array>
   - x_0: initial state of the system at t=0
          <ROW vector of type numpy array> or scalar
   - U: sequence of control inputs that are applied to the system one after 
         another. Each control input is applied with duration of delta_t, 
         where it is assumed that the control input is constant during that 
         time
         <numpy array where every row represents a control input that is 
         applied at a certain time. The row number corresponds to the number 
         of the integration interval when the control input is applied>
   - delta_t: length of a single integration interval. The total simulated
         time is equal to delta_t * U.shape[0]
         <positive float>
    - my_outFunc: output function of the form y(t) = g(x(t), u(t)) that maps 
         the state and control input to the output
          <function handle to function g. The inputs and outputs of 
         function g must be scalars or ROW vectors of type numpy 
         array>
    
Output Parameter:
   - X: Trajectory of system states
         <numpy array where every row corresponds to the (multidimensional) 
         state of the system at the beginning of a certain time step. The time 
         step is given by the index of that row>
   - Y: Trajectory of system outputs
         <numpy array where every row corresponds to the (multidmensional) 
         output of the system for a certain time step. The time step is 
         given by the index of that row>
   - T: Time line that corresponds to X and Y
"""


def nlsim(my_ode, x_0, U, delta_t, my_out_func):
    # init state, state trajectory, and output trajectory
    x = x_0
    X = [x_0]
    Y = [my_out_func(x_0, U[0:])]

    # forward simulation of ODE with given control sequence
    for u in U:
        # calculate state at end of time delta_t using Runge-Kutta integrator
        x = rk(my_ode, delta_t, x, u)

        # append state to state trajectory
        X.append(x)

        # append output value to output trajectory
        Y.append(my_out_func(x, u))

    # create array with used time line for convenient plotting afterwards
    T = np.linspace(0, delta_t * U.shape[0], U.shape[0] + 1)

    # Before returning, convert X and Y from lists to arrays
    return (np.array(X), np.array(Y), T)
