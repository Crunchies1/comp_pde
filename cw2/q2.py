import numpy as np
import matplotlib.pyplot as plt
import time

def grad_yb_aerofoil(x, tao):
    '''
    input: 
        x: x coordinate, float
        t: maximum thickness of yb, float
    output:
        grad of aerofoil at corresponding x coordinate
    '''
    return (tao * (1 - 2 * x)) / np.sqrt(x * (1 - x))

def finite_diff_compressible_fluid(q, s, r, u, M, step_size):
    # Create mesh grid where x step and y step same
    x_range = np.arange(-q, s + step_size, step_size)
    y_range = np.arange(0, r + step_size, step_size)
    X, Y = np.meshgrid(x_range, y_range)

    # Initial condition
    U = np.zeros_like(X)
    U[:, 0] = -u
    U[:, -1] = u
    m, n = U.shape
    P = np.random.rand(m, n)
    P[:, 0] = 0
    P[:, -1] = 0
    P[0, :] = 0
    P[-1, :] = 0
    P = P + 1

    U_prev = np.ones_like(U)
    while np.linalg.norm(U_prev - U) > 1e-5:
        U_prev = 1.0 * U
        for i in range(m):
            for j in range(n):
                # Top left corner
                if i == 0 and j == 0:
                    # U[i, j] = 1/2 * (U[i + 1, j] + U[i, j + 1])
                    U[i, j] = -u
                    P[i, j] = 1
                    
                # Top right corner
                elif i == 0 and j == n - 1:
                    # U[i, j] = 1/2 * (U[i + 1, j] + U[i, j - 1])
                    U[i, j] = u
                    P[i, j] = 1

                # Bottom left corner
                elif i == m - 1 and j == 0:
                    # U[i, j] = 1/2 * (U[i - 1, j] + U[i, j + 1])
                    U[i, j] = -u
                    P[i, j] = 1

                # Bottom right corner
                elif i == m - 1 and j == n - 1:
                    # U[i, j] = 1/2 * (U[i - 1, j] + U[i, j - 1])
                    U[i, j] = u
                    P[i, j] = 1

                # For any point on upper boundary
                elif i == 0 and (j > 0 and j < (n - 1)):
                    dpdx = P[i, j + 1] - P[i, j - 1]
                    dpdy = 2 * P[i + 1, j]
                    U_x_adj = U[i, j + 1] + U[i, j - 1]
                    U_y_adj = 2 * U[i + 1, j]
                    dphidx = U[i, j + 1] - U[i, j - 1]
                    dphidy = 0
                    P[i, j] = 1
                    if dpdx == 0 and dpdy == 0:
                        U[i, j] = 0
                    else:
                        U[i, j] = (1/(16 * P[i, j])) * ((step_size * 2 * dpdx) + (dpdx * dphidx) + (dpdy * dphidy)) + 1/4 * (U_x_adj + U_y_adj)

                # For any point on left boundary
                elif (i > 0 and i < (m - 1)) and j == 0:
                    # U[i, j] = 1/4 * (U[i + 1, j] + U[i - 1, j] + 2 * U[i, j + 1])
                    U[i, j] = -u
                    P[i, j] = 1

                # For any point on right boundary
                elif (i > 0 and i < (m - 1)) and j == n - 1:
                    # U[i, j] = 1/4 * (U[i + 1, j] + U[i - 1, j] + 2 * U[i, j - 1])
                    U[i, j] = u
                    P[i, j] = 1

                # For any point on the non-aerofoil bottom boundary
                elif i == m - 1 and (x_range[j] < 0.01 or x_range[j] > 0.99):
                    dpdx = P[i, j + 1] - P[i, j - 1]
                    dpdy = 2 * P[i - 1, j]
                    U_x_adj = U[i, j + 1] + U[i, j - 1]
                    U_y_adj = 2 * U[i - 1, j]
                    dphidx = U[i, j + 1] - U[i, j - 1]
                    dphidy = 0
                    P[i, j] = 1
                    if dpdx == 0 and dpdy == 0:
                        U[i, j] = 0
                    else:
                        U[i, j] = (1/(16 * P[i, j])) * ((step_size * 2 * dpdx) + (dpdx * dphidx) + (dpdy * dphidy)) + 1/4 * (U_x_adj + U_y_adj)

                # For any point on the aerofoil bottom boundary
                elif i == m - 1:
                    grad_foil = grad_yb_aerofoil(x_range[j], 0.05)
                    dpdx = P[i, j + 1] - P[i, j - 1]
                    dpdy = 2 * P[i - 1, j]
                    U_x_adj = U[i, j + 1] + U[i, j - 1]
                    U_y_adj = 2 * U[i - 1, j] - (2 * step_size + U[i, j + 1] - U[i, j - 1]) * grad_foil
                    dphidx = U[i, j + 1] - U[i, j - 1]
                    dphidy = (2 * step_size + U[i, j + 1] - U[i, j - 1]) * grad_foil
                    P[i, j] = 1
                    if dpdx == 0 and dpdy == 0:
                        U[i, j] = 0
                    else:
                        U[i, j] = (1/(16 * P[i, j])) * ((step_size * 2 * dpdx) + (dpdx * dphidx) + (dpdy * dphidy)) + 1/4 * (U_x_adj + U_y_adj)

                # For any point in the grid not on the edge
                else:
                    dpdx = P[i, j + 1] - P[i, j - 1]
                    dpdy = P[i + 1, j] - P[i - 1, j]
                    U_x_adj = U[i, j + 1] + U[i, j - 1]
                    U_y_adj = U[i + 1, j] + U[i - 1, j]
                    dphidx = U[i, j + 1] - U[i, j - 1]
                    dphidy = U[i + 1, j] - U[i - 1, j]
                    P[i, j] = ((1 - (0.2 * M ** 2) * ((U_x_adj / step_size) + (U_x_adj / (2 * step_size)) ** 2 + (U_y_adj / (2 * step_size) ** 2))) ** (5 / 2))
                    if dpdx == 0 and dpdy == 0:
                        U[i, j] = 0
                    else:
                        U[i, j] = (1/(16 * P[i, j])) * ((step_size * 2 * dpdx) + (dpdx * dphidx) + (dpdy * dphidy)) + 1/4 * (U_x_adj + U_y_adj)
        
    return np.flip(U, axis=0), X, Y, np.flip(P)

# q, s, r, u, M, step_size
U, X, Y, P = finite_diff_compressible_fluid(1, 2, 1, 1, 0.04, 0.1)
print(U)
print(P)

velocity_y, velocity_x = np.gradient(U)

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