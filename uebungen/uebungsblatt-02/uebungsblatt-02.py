import numpy as np

a = np.array([[1, 2],
              [3, 4]])
b = np.arange(1, 6)
c = np.eye(3)
d = np.ones((2, 5))
e = np.zeros(a.shape)

a_dot = np.array([[0, 0],
                 [0, 0]])
a_multiply = np.array([[0, 0],
                       [0, 0]])
np.dot(a, a, a_dot)
np.multiply(a, a, a_multiply)

print("Matrixprodukt aus AA:\n", a_dot, "\n")
print("Elementweise Multiplikation von A*A:\n", a_multiply, "\n")

f = a[:, 0]

print("Erste Spalte von a: \n", f, "\n")
print("A@f:\n", a@f, "\n\nf@A:\n", f@a, "\n\nf.T@A:\n", f.T@a, "\n")


def myfirstfunction(a, b):
    print("Hello {} World".format(a))
    print(np.sin(b))
    return


myfirstfunction("Python", 0)
myfirstfunction("numerical", 90)
