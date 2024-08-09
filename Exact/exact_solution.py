import os
import numpy as np
from scipy.special import roots_hermite
import matplotlib.pyplot as plt


def burgers_viscous_time_exact1(nu, vxn, vx, vtn, vt):
    """
    Evaluate the solution to the Burgers equation.

    Parameters:
    nu  - viscosity
    vxn - number of spatial grid points
    vx  - spatial grid points (numpy array)
    vtn - number of time grid points
    vt  - time grid points (numpy array)

    Returns:
    vu  - solution of the Burgers equation at each space and time grid point
    """
    # Create the directory if it doesn't exist
    directory = "CalculatedExactSolutions"
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Create a unique filename based on the parameters
    filename = os.path.join(directory, f"burgers_solution_nu_{nu:.6f}_vxn_{vxn}_vtn_{vtn}.npz")

    #If solutin has already been calculated just load the data
    # Check if the file already exists
    if os.path.exists(filename):
        data = np.load(filename)
        vu = data['vu']
        return vu

    qn = 400

    # Compute the rule using Hermite quadrature
    qx, qw = roots_hermite(qn)

    # Evaluate U(X,T) for later times
    vu = np.zeros((vxn, vtn))

    for vti in range(vtn):
        if vt[vti] == 0.0:
            vu[:, vti] = -np.sin(np.pi * vx)
        else:
            for vxi in range(vxn):
                top = 0.0
                bot = 0.0

                for qi in range(qn):
                    c = 2.0 * np.sqrt(nu * vt[vti])

                    top -= (qw[qi] * c * np.sin(np.pi * (vx[vxi] - c * qx[qi])) *
                            np.exp(-np.cos(np.pi * (vx[vxi] - c * qx[qi])) /
                                   (2.0 * np.pi * nu)))

                    bot += (qw[qi] * c *
                            np.exp(-np.cos(np.pi * (vx[vxi] - c * qx[qi])) /
                                   (2.0 * np.pi * nu)))

                vu[vxi, vti] = top / bot

    # Save the result to a file
    np.savez(filename, vu=vu)

    return vu
'''

# Parameters
nu = 0.01 / np.pi
vxn = 101
vx = np.linspace(-1.0, 1.0, vxn)
vtn = 100
vt = np.linspace(0.0, 1.0, vtn)

# Compute the solution
vu = burgers_viscous_time_exact1(nu, vxn, vx, vtn, vt)

#-------------------------- 3D Plot
# Plot the results
X, T = np.meshgrid(vx, vt, indexing='ij')
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(X, T, vu, cmap='viridis')
ax.set_xlabel('Space (x)')
ax.set_ylabel('Time (t)')
ax.set_zlabel('U(x,t)')
ax.set_title('Solution to the Burgers Equation')
plt.show()
#--------------------

# Plot the results
time_indices = np.linspace(0, vtn - 1, 10, dtype=int)
fig, axes = plt.subplots(5, 2, figsize=(15, 20))
axes = axes.flatten()

for i, ax in enumerate(axes):
    t_idx = time_indices[i]
    ax.plot(vx, vu[:, t_idx], label=f't = {vt[t_idx]:.2f}s')
    ax.set_xlabel('Space (x)')
    ax.set_ylabel('U(x,t)')
    ax.legend()
    ax.grid()

plt.tight_layout()
plt.show()

'''