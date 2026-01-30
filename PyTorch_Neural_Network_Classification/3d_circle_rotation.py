import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Circle parameters
r = 1  # radius
theta = np.linspace(0, 2 * np.pi, 100)

# Circle coordinates in XY plane initially
x = r * np.cos(theta)
y = r * np.sin(theta)
z = np.zeros_like(theta)

# Create 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

# Rotation angles (in radians)
phi_vals = np.linspace(0, 2 * np.pi, 60)  # rotation around X axis

for phi in phi_vals:
    # Rotate around X axis
    z_rot = z * np.cos(phi) - y * np.sin(phi)
    y_rot = z * np.sin(phi) + y * np.cos(phi)

    ax.clear()
    ax.plot(x, y_rot, z_rot)

    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_zlim(-1.5, 1.5)

    plt.pause(0.05)

plt.show()
