import matplotlib.pyplot as plt
import numpy as np

x1 = np.arange(0, 5, 0.1)
y1 = np.sin(x1)
x2 = np.linspace(1, 10, 20, True)
y2 = np.cos(x2)

plt.plot(x1, y1, 'b^')
