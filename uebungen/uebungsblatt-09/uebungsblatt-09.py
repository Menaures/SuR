import control
import numpy
import matplotlib.pyplot as plot

def system():
    """
    This function calculates the transferfunction of series cirucit of
    Resistance R1 with capacity C1 and C2 || R2. Aswell the system.
    state representation.
    """
    R1 = 20e3
    C1 = 15e-9
    C2 = 560e-12
    R2 = 10e3
    T1 = 1 / (R1 * C1)
    T2 = 1 / (R2 * C2)
    T3 = 1 / (R1 * C2)
    numerator_out = [T1, 0]
    dominator_out = [1, (T1 + T2 + T3), T1 * T2]
    
    return numerator_out, dominator_out

omega = numpy.arange(0, 10e3, 50)

system = control.tf(system()[0], system()[1])
control.bode_plot(system, dB = True)
print(control.pole(system))
plot.show()
