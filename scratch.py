import math

import numpy as np
from matplotlib import pyplot as plt

plt.rcParams["figure.figsize"] = [7.50, 3.50]
plt.rcParams["figure.autolayout"] = True

def f(x):
   return np.sin(x) + x + x * np.sin(x)

x = np.linspace(-10, 10, 100)

plt.plot(x, f(x), 'red')

plt.show()

print(math.exp(1))