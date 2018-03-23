import numpy as ny

a = ny.random.randn(5,1)
b = a.T * a
c = a * a.T
print(b)
print(c)

