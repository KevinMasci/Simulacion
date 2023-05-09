import random
from numpy import log as ln
import matplotlib.pyplot as plt

# Uniforme continua
def uniforme(a, b):
    r = random.random()
    num = a + (b-a) * r
    return num

# Exponencial
def exponencial(lamb):
    r = random.random()
    num = -lamb * ln(r)
    return num

# Gamma
def normal(ex, stdx):
    sum = 0
    for i in range (30):
        r = random.random()
        sum += r
    x = stdx * (sum - 15.0) + ex
    return x

dist = [normal(10, 25) for _ in range(100)]
print(dist)
plt.hist(dist)
plt.show()