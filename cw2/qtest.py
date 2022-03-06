import numpy as np
import matplotlib.pyplot as plt

def grad_yb_aerofoil(x, tao):
    '''
    input: 
        x: x coordinate, float
        t: maximum thickness of yb, float
    output:
        grad of aerofoil at corresponding x coordinate
    '''
    return (tao * (1 - 2 * x)) / np.sqrt(x * (1 - x))

def finite_diff_laplace(q, s, r, u, step_size):
    # Create mesh grid where x step and y step same
    x_range = np.arange(-q, s + step_size, step_size)
    y_range = np.arange(0, r + step_size, step_size)
    X, Y = np.meshgrid(x_range, y_range)

    # Initial condition
    U = np.zeros_like(X)
    m, n = U.shape
    U[:, 0] = -u 
    U[:, -1] = u 
    P = np.zeros_like(U)


    U_prev = np.ones_like(U)
    while np.linalg.norm(U_prev - U) > 1e-8:
        U_prev = 1.0 * U
        for i in range(m):
            for j in range(n):
                # Top left corner
                if i == 0 and j == 0:
                    # U[i, j] = 1/2 * (U[i + 1, j] + U[i, j + 1])
                    U[i, j] = -u
                    
                # Top right corner
                elif i == 0 and j == n - 1:
                    # U[i, j] = 1/2 * (U[i + 1, j] + U[i, j - 1])
                    U[i, j] = u

                # Bottom left corner
                elif i == m - 1 and j == 0:
                    # U[i, j] = 1/2 * (U[i - 1, j] + U[i, j + 1])
                    U[i, j] = -u

                # Bottom right corner
                elif i == m - 1 and j == n - 1:
                    # U[i, j] = 1/2 * (U[i - 1, j] + U[i, j - 1])
                    U[i, j] = u

                # For any point on upper boundary
                elif i == 0 and (j > 0 and j < (n - 1)):
                    U[i, j] = 1/4 * (2 * U[i + 1, j] + U[i, j + 1] + U[i, j - 1])

                # For any point on left boundary
                elif (i > 0 and i < (m - 1)) and j == 0:
                    # U[i, j] = 1/4 * (U[i + 1, j] + U[i - 1, j] + 2 * U[i, j + 1])
                    U[i, j] = -u

                # For any point on right boundary
                elif (i > 0 and i < (m - 1)) and j == n - 1:
                    # U[i, j] = 1/4 * (U[i + 1, j] + U[i - 1, j] + 2 * U[i, j - 1])
                    U[i, j] = u

                # For any point on the non-aerofoil bottom boundary
                elif i == m - 1 and (x_range[j] < 0.01 or x_range[j] > 0.99):
                    U[i, j] = 1/4 * (2 * U[i - 1, j] + U[i, j - 1] + U[i, j + 1])  

                # For any point on the aerofoil bottom boundary
                elif i == m - 1:
                    grad_foil = grad_yb_aerofoil(x_range[j], 0.05)
                    U[i, j] = 1/4 * (U[i, j + 1] * (1 - grad_foil) + U[i, j - 1] * (1 + grad_foil) + 2 * (U[i - 1, j] - step_size * grad_foil))

                # For any point in the grid not on the edge
                else:
                    U[i, j] = 1/4 * (U[i - 1, j] + U[i + 1, j] + U[i, j - 1] + U[i, j + 1])

    return np.flip(U, axis=0), X, Y

U, X, Y = finite_diff_laplace(1, 2, 1, 1, 0.1)

velocity_y, velocity_x = np.gradient(U)

print(U)

plt.plot(X[0], velocity_x[0], 'go--', linewidth=2, markersize=12)
plt.show()

colorinterpolation = 50
colourMap = plt.cm.jet #you can try: colourMap = plt.cm.coolwarm

# Set Colorbar
plt.contourf(X, Y, U, colorinterpolation, cmap=colourMap)
plt.colorbar()
plt.show()


plt.quiver(X, Y, velocity_x, velocity_y)
plt.axis('equal')
plt.show()