import numpy as np
import matplotlib.pyplot as plt

theta = np.linspace(0, 2 * np.pi, 100)
x = np.cos(theta)
y = np.sin(theta)

plt.plot(x, y)

for t in np.linspace(0, 2 * np.pi, 60):
    px = np.cos(t)
    py = np.sin(t)
    plt.scatter(px, py)
    plt.pause(0.1)
    plt.clf()
    plt.plot(x, y)

plt.show()
