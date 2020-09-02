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
import control as ctrl

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
   length delta_t is performed for each contrtol input in the provided 
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


def nlsim(my_ode, x_0, U, delta_t, my_outFunc):
    # init state, state trajectory, and output trajectory
    x = x_0
    X = [x_0]
    Y = [my_outFunc(x_0, U[0:, ])]

    # forward simulation of ODE with given control sequence
    for u in U:
        # calculate state at end of time delta_t using Runge-Kutta integrator
        x = rk(my_ode, delta_t, x, u)

        # append state to state trajectory
        X.append(x)

        # append output value to output trajectory
        Y.append(my_outFunc(x, u))

    # create array with used time line for convenient plotting afterwards
    T = np.linspace(0, delta_t * U.shape[0], U.shape[0] + 1)

    # Before returning, convert X and Y from lists to arrays
    return (np.array(X), np.array(Y), T)


def delay(sys, T):
    """========================================================================
    Function: delay() 
        approximates the given delay with a PT100 element and concatenates it 
        with the given system.
    
    Synopsis:
        sys1 = delay(sys,T)
    
    Input Parameter:
        - sys: Control module LTI-System in transfer function or state space
            formulation. The delay will be added to this system.
        - T: Delay time that will be added to this system.
    
    Output Parameter:
        - sys1: The delayed input system.
    ========================================================================"""
    # define the PT100 element
    n = 100
    T = float(T) / n
    A = (np.diag([-1] * n, 0) + np.diag([1] * (n - 1), -1)) / T
    B = np.vstack([1, [[0]] * (n - 1)]) / T
    C = np.hstack([[0] * (n - 1), 1])
    D = 0
    # concatenate it with the system
    return ctrl.series(sys, ctrl.ss(A, B, C, D))


def unknown_step(Kp=None, Ki=None, Kd=None):
    """========================================================================
    Function: unknown_step() 
        generates a step response of the unkown system used in exercise 3 of 
        sheet 10. The response could either be openloop or with a PID 
        controller. For the PID controller you have to specify the Kp, Ki and 
        Kd gains.
    
    Synopsis:
        y,t = unknown_step()            # openloop
        y,t = unknown_step(Kp,Ki,Kd)    # with PID controller
    
    Input Parameter:
        - Kp: Proportional gain
        - Ki: Integral gain
        - Kd: Derivative gain
    
    Output Parameter:
        - y: The step response of the system
        - t: The time grid of the step response
        where y is actually t and vice versa
    ========================================================================"""
    t = np.linspace(0, 50, 100)
    # the system is unknown, DONT read it!
    sys = 5000 * ctrl.tf([1, 12, 32],
                         [1, 29, 829, 8939, 79418, 187280, 116000])
    sys = delay(sys, 2.1)
    # openloop if no K is specified
    if Kp == None or Ki == None or Kd == None:
        y, t = ctrl.step_response(sys, t)
        return y, t
    # closedloop PID 
    else:
        PID = ctrl.tf([Kd, Kp, Ki], [1e-10, 1, 0])
        # the 1e-10 is a bit nasty but otherwise it cannot be turned into a 
        # state space model, which is needed for the feedback to work proper.
        PIDss = ctrl.ss(PID)
        sysPID = ctrl.feedback(ctrl.series(PIDss, sys), 1)
        y, t = ctrl.step_response(sysPID, t)
        return y, t
