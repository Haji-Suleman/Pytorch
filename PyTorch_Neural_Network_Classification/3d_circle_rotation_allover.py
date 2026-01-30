import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Circle parameters
r = 1
theta = np.linspace(0, 2 * np.pi, 100)

# Initial circle coordinates in XY plane
x = r * np.cos(theta)
y = r * np.sin(theta)
z = np.zeros_like(theta)

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

# Rotation angles for all axes
phi_vals = np.linspace(0, 2 * np.pi, 120)  # rotate X
theta_vals = np.linspace(0, 2 * np.pi, 120)  # rotate Y
psi_vals = np.linspace(0, 2 * np.pi, 120)  # rotate Z

for phi, th, psi in zip(phi_vals, theta_vals, psi_vals):
    # Rotation matrices
    # Rotate around X
    y_rot = y * np.cos(phi) - z * np.sin(phi)
    z_rot = y * np.sin(phi) + z * np.cos(phi)

    # Rotate around Y
    x_rot = x * np.cos(th) + z_rot * np.sin(th)
    z_rot2 = -x * np.sin(th) + z_rot * np.cos(th)

    # Rotate around Z
    x_rot2 = x_rot * np.cos(psi) - y_rot * np.sin(psi)
    y_rot2 = x_rot * np.sin(psi) + y_rot * np.cos(psi)

    ax.clear()
    ax.plot(x_rot2, y_rot2, z_rot2, color="blue", linewidth=2)

    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_zlim(-1.5, 1.5)

    plt.pause(0.05)

plt.show()
