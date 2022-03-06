import numpy as np
import matplotlib.pyplot as plt
import time

def yb_aerofoil(x, tao):
    '''
    input: 
        x: x coordinate, float
        t: maximum thickness of yb, float
    output:
        half thickness of airfoil at corresponding x coordinate
    '''
    return 2 * tao * np.sqrt(x * (1 - x))

def grad_yb_aerofoil(x, tao):
    '''
    input: 
        x: x coordinate, float
        t: maximum thickness of yb, float
    output:
        grad of aerofoil at corresponding x coordinate
    '''
    return (tao * (1 - 2 * x)) / np.sqrt(x * (1 - x))

def finite_diff_laplace(q, s, r, u, step_size, type = 'GS'):
    # Create mesh grid where x step and y step same
    x_range = np.arange(-q, s + step_size, step_size)
    y_range = np.arange(0, r + step_size, step_size)
    X, Y = np.meshgrid(x_range, y_range)

    # Initial condition
    U = np.zeros_like(X)
    U[:, 0] = -u
    U[:, -1] = u
    m, n = U.shape
    loops = 0

    if type == 'Jacobi':
        U_cur = np.zeros_like(X)
        U_cur[:, 0] = -u
        U_cur[:, -1] = u
        U = np.ones_like(U_cur)

        while np.linalg.norm(U_cur - U) > 1e-5:
            U = 1.0 * U_cur
            for i in range(m):
                for j in range(n):
                    # Top left corner
                    if i == 0 and j == 0:
                        # U_cur[i, j] = 1/2 * (U[i + 1, j] + U[i, j + 1])
                        U_cur[i, j] = -u
                        
                    # Top right corner
                    elif i == 0 and j == n - 1:
                        # U_cur[i, j] = 1/2 * (U[i + 1, j] + U[i, j - 1])
                        U_cur[i, j] = u

                    # Bottom left corner
                    elif i == m - 1 and j == 0:
                        # U_cur[i, j] = 1/2 * (U[i - 1, j] + U[i, j + 1])
                        U_cur[i, j] = -u

                    # Bottom right corner
                    elif i == m - 1 and j == n - 1:
                        # U_cur[i, j] = 1/2 * (U[i - 1, j] + U[i, j - 1])
                        U_cur[i, j] = u

                    # For any point on upper boundary
                    elif i == 0 and (j > 0 and j < (n - 1)):
                        U_cur[i, j] = 1/4 * (2 * U[i + 1, j] + U[i, j + 1] + U[i, j - 1])

                    # For any point on left boundary
                    elif (i > 0 and i < (m - 1)) and j == 0:
                        # U_cur[i, j] = 1/4 * (U[i + 1, j] + U[i - 1, j] + 2 * U[i, j + 1])
                        U_cur[i, j] = -u

                    # For any point on right boundary
                    elif (i > 0 and i < (m - 1)) and j == n - 1:
                        # U_cur[i, j] = 1/4 * (U[i + 1, j] + U[i - 1, j] + 2 * U[i, j - 1])
                        U_cur[i, j] = u

                    # For any point on the non-aerofoil bottom boundary
                    elif i == m - 1 and (x_range[j] < 0.01 or x_range[j] > 0.99):
                        U_cur[i, j] = 1/4 * (2 * U[i - 1, j] + U[i, j - 1] + U[i, j + 1])  

                    # For any point on the aerofoil bottom boundary
                    elif i == m - 1:
                        grad_foil = grad_yb_aerofoil(x_range[j], 0.05)
                        U_cur[i, j] = 1/4 * (U[i, j + 1] * (1 - grad_foil) + U[i, j - 1] * (1 + grad_foil) + 2 * (U[i - 1, j] - step_size * grad_foil))

                    # For any point in the grid not on the edge
                    else:
                        U_cur[i, j] = 1/4 * (U[i - 1, j] + U[i + 1, j] + U[i, j - 1] + U[i, j + 1])
            loops += 1

    if type == 'GS':
        U_prev = np.ones_like(U)
        while np.linalg.norm(U_prev - U) > 1e-5:
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
            loops += 1

    return np.flip(U, axis=0), X, Y, loops

def finite_diff_laplace_SOR(q, s, r, u, step_size, alpha):
    # Create mesh grid where x step and y step same
    x_range = np.arange(-q, s + step_size, step_size)
    y_range = np.arange(0, r + step_size, step_size)
    X, Y = np.meshgrid(x_range, y_range)

    # Initial condition
    U = np.zeros_like(X)
    U[:, -1] = u
    m, n = U.shape
    loops = 0

    U_prev = np.ones_like(U)
    while np.linalg.norm(U_prev - U) > 1e-5:
        U_prev = 1.0 * U
        for i in range(m):
            for j in range(n):
                # Top left corner
                if i == 0 and j == 0:
                    # U[i, j] = U[i, j] + alpha * (1/2 * (U[i + 1, j] + U[i, j + 1]) - U[i, j])
                    U[i, j] = -u
                    
                # Top right corner
                elif i == 0 and j == n - 1:
                    # U[i, j] = U[i, j] + alpha * (1/2 * (U[i + 1, j] + U[i, j - 1]) - U[i, j])
                    U[i, j] = u

                # Bottom left corner
                elif i == m - 1 and j == 0:
                    # U[i, j] = U[i, j] + alpha * (1/2 * (U[i - 1, j] + U[i, j + 1]) - U[i, j])
                    U[i, j] = -u

                # Bottom right corner
                elif i == m - 1 and j == n - 1:
                    # U[i, j] = U[i, j] + alpha * (1/2 * (U[i - 1, j] + U[i, j - 1]) - U[i, j])
                    U[i, j] = u

                # For any point on upper boundary
                elif i == 0 and (j > 0 and j < (n - 1)):
                    U[i, j] = U[i, j] + alpha * (1/4 * (2 * U[i + 1, j] + U[i, j + 1] + U[i, j - 1]) - U[i, j])

                # For any point on left boundary
                elif (i > 0 and i < (m - 1)) and j == 0:
                    # U[i, j] = U[i, j] + alpha * (1/4 * (U[i + 1, j] + U[i - 1, j] + 2 * U[i, j + 1]) - U[i, j])
                    U[i, j] = -u

                # For any point on right boundary
                elif (i > 0 and i < (m - 1)) and j == n - 1:
                    # U[i, j] = U[i, j] + alpha * (1/4 * (U[i + 1, j] + U[i - 1, j] + 2 * U[i, j - 1]) - U[i, j])
                    U[i, j] = u

                # For any point on the non-aerofoil bottom boundary
                elif i == m - 1 and (x_range[j] < 0.01 or x_range[j] > 0.99):
                    U[i, j] = U[i, j] + alpha * (1/4 * (2 * U[i - 1, j] + U[i, j - 1] + U[i, j + 1]) - U[i, j])

                # For any point on the aerofoil bottom boundary
                elif i == m - 1 and (0 < x_range[j] and x_range[j] < 1):
                    grad_foil = grad_yb_aerofoil(x_range[j], 0.05)
                    U[i, j] = U[i, j] + alpha * (1/4 * (U[i, j + 1] * (1 - grad_foil) + U[i, j - 1] * (1 + grad_foil) + 2 * (U[i - 1, j] - step_size * grad_foil)) - U[i, j])

                # For any point in the grid not on the edge
                else:
                    U[i, j] = U[i, j] + alpha * (1/4 * (U[i - 1, j] + U[i + 1, j] + U[i, j - 1] + U[i, j + 1]) - U[i, j])
        loops += 1

    return np.flip(U, axis=0), X, Y, loops

def dxdy_shifted(x, tao):
    if (x == 0.5):
        return 0
    return np.sqrt(x - x ** 2) / (tao * (1 - 2 * x))

def finite_diff_laplace_shifted_SOR(q, s, r, u, step_size, alpha):
    # Create mesh grid where x step and y step same
    x_range = np.arange(-q, s + step_size, step_size)
    n_range = np.arange(0, r + step_size, step_size)
    X, N = np.meshgrid(x_range, n_range)

    # Initial condition
    U = np.zeros_like(X)
    U[:, -1] = u
    m, n = U.shape

    U_prev = np.ones_like(U)
    while np.linalg.norm(U_prev - U) > 1e-5:
        U_prev = 1.0 * U
        for i in range(m):
            for j in range(n):
                # Top left corner
                if i == 0 and j == 0:
                    # U[i, j] = U[i, j] + alpha * (1/2 * (U[i + 1, j] + U[i, j + 1]) - U[i, j])
                    U[i, j] = -u
                    
                # Top right corner
                elif i == 0 and j == n - 1:
                    # U[i, j] = U[i, j] + alpha * (1/2 * (U[i + 1, j] + U[i, j - 1]) - U[i, j])
                    U[i, j] = u

                # Bottom left corner
                elif i == m - 1 and j == 0:
                    # U[i, j] = U[i, j] + alpha * (1/2 * (U[i - 1, j] + U[i, j + 1]) - U[i, j])
                    U[i, j] = -u

                # Bottom right corner
                elif i == m - 1 and j == n - 1:
                    # U[i, j] = U[i, j] + alpha * (1/2 * (U[i - 1, j] + U[i, j - 1]) - U[i, j])
                    U[i, j] = u

                # For any point on upper boundary
                elif i == 0 and (j > 0 and j < (n - 1)):
                    U[i, j] = U[i, j] + alpha * (1/4 * (2 * U[i + 1, j] + U[i, j + 1] + U[i, j - 1]) - U[i, j])

                # For any point on left boundary
                elif (i > 0 and i < (m - 1)) and j == 0:
                    # U[i, j] = U[i, j] + alpha * (1/4 * (U[i + 1, j] + U[i - 1, j] + 2 * U[i, j + 1]) - U[i, j])
                    U[i, j] = -u

                # For any point on right boundary
                elif (i > 0 and i < (m - 1)) and j == n - 1:
                    # U[i, j] = U[i, j] + alpha * (1/4 * (U[i + 1, j] + U[i - 1, j] + 2 * U[i, j - 1]) - U[i, j])
                    U[i, j] = u

                # For any point on the non-aerofoil bottom boundary
                elif i == m - 1 and (x_range[j] < 0.01 or x_range[j] > 0.99):
                    U[i, j] = U[i, j] + alpha * (1/4 * (2 * U[i - 1, j] + U[i, j - 1] + U[i, j + 1]) - U[i, j])

                # For any point on the aerofoil bottom boundary
                elif i == m - 1:
                    r = (1 + dxdy_shifted(x_range[j], 0.05) ** 2)
                    U[i, j] = U[i, j] + alpha * (1/4 * (U[i, j + 1] * (1 + r) + U[i, j - 1] * (1 - r) + 2 * (U[i - 1, j])) - U[i, j])

                # For any point in the grid not in the aerofoil range
                elif (x_range[j] < 0.01 or x_range[j] > 0.99):
                    U[i, j] = U[i, j] + alpha * (1/4 * (U[i - 1, j] + U[i + 1, j] + U[i, j - 1] + U[i, j + 1]) - U[i, j])

                # For any other point in grid
                else:
                    r = (1 + dxdy_shifted(x_range[j], 0.05) ** 2)
                    U[i, j] = U[i, j] + alpha * (1/(2 + 2 * r) * (U[i - 1, j] + U[i + 1, j] + r * (U[i, j - 1] + U[i, j + 1])) - U[i, j])

    return np.flip(U, axis=0), X, N


def time_and_plots(type = 'GS'):
    start = time.time()
    if type == 'GS':
        # q, s, r, u, step_size, type
        U, X, Y, loops = finite_diff_laplace(1, 4, 0.75, 3.5, 0.05, 'GS')
    elif type == 'Jacobi':
        U, X, Y, loops = finite_diff_laplace(1, 4, 0.75, 3.5, 0.05, 'Jacobi')
    elif type == 'SOR':
        # q, s, r, u, step_size, alpha
        U, X, Y, loops = finite_diff_laplace_SOR(1, 4, 0.75, 3.5, 0.05, 1)
    elif type == 'Shifted':
        U, X, Y, loops = finite_diff_laplace_shifted_SOR(1, 2, 1, 1, 0.1, 1)
    end = time.time()

    # Show convergence times
    computation_time = end - start
    print(computation_time, loops)

    velocity_y, velocity_x = np.gradient(U)

    # Plot U_surf
    plt.plot(X[0], velocity_x[0], 'go--', linewidth=1, markersize=4)
    plt.xlabel("x")
    plt.ylabel("Partial derivative of phi with respect to x")
    plt.title("U_surf against x at y = 0")
    plt.show()

    # Plot V_surf
    plt.plot(X[0], velocity_y[0], 'go--', linewidth=1, markersize=4)
    plt.xlabel("x")
    plt.ylabel("Partial derivative of phi with respect to y")
    plt.title("U_surf_y against x at y = 0")
    plt.show()

    # Plot colour graph
    colorinterpolation = 50
    colourMap = plt.cm.jet
    plt.contourf(X, Y, U, colorinterpolation, cmap=colourMap)
    plt.colorbar()
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Velocity potential at (x, y)")
    plt.show()

    # Plot quiver graph
    plt.quiver(X, Y, velocity_x, velocity_y)
    plt.axis('equal')
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Velocity at (x, y)")
    plt.show()

time_and_plots('GS')
time_and_plots('Jacobi')
time_and_plots('SOR')

