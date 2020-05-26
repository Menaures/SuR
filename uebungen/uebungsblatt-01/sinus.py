import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 3*np.pi)
y = np.sin(x)
plt.figure()
plt.plot(x, y)
plt.xlabel("x")
plt.ylabel("y")
plt.title("Sinus")
plt.legend(["f(x) = sin(x)"])
plt.show()
