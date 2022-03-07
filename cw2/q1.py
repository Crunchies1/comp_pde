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

# 0 < x < 1
def finite_diff_laplace_shifted_SOR(r, u, step_size, alpha, tao):
    # Create mesh grid where x step and y step same
    x_range = np.arange(0.01, 1, step_size)
    n_range = np.arange(0.01, r, step_size)
    X, N = np.meshgrid(x_range, n_range)

    # Initial condition
    U = np.zeros_like(X)
    U[:, -1] = u
    U[:, 1] = -u
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
                    dndx = (tao * (2 * x_range[j] - 1)) / np.sqrt(x_range[j] - x_range[j] ** 2)
                    d2ndx2 = tao / (2 * (x_range[j] - x_range[j] ** 2) ** (3/2))
                    x_adj = U[i, j + 1] + U[i, j - 1]
                    n_adj = 2 * U[i + 1, j]
                    dphidn = 2 * U[i + 1, j]
                    diag_sum = 0
                    U[i, j] = U[i, j] + alpha * (((x_adj + ((diag_sum * dndx) / 2) + (n_adj * ((dndx ** 2) + 1)) + ((step_size/2) * (dphidn) * (d2ndx2))) / (2 * (2 + dndx ** 2))) - U[i, j])
          
                # For any point on left boundary
                elif (i > 0 and i < (m - 1)) and j == 0:
                    # U[i, j] = U[i, j] + alpha * (1/4 * (U[i + 1, j] + U[i - 1, j] + 2 * U[i, j + 1]) - U[i, j])
                    U[i, j] = -u

                # For any point on right boundary
                elif (i > 0 and i < (m - 1)) and j == n - 1:
                    # U[i, j] = U[i, j] + alpha * (1/4 * (U[i + 1, j] + U[i - 1, j] + 2 * U[i, j - 1]) - U[i, j])
                    U[i, j] = u

                # For any point on the aerofoil bottom boundary
                elif i == m - 1 and (j > 1 and j < (n - 2)):
                    grad_foil = grad_yb_aerofoil(x_range[j], 0.05)
                    dndx = (tao * (2 * x_range[j] - 1)) / np.sqrt(x_range[j] - x_range[j] ** 2)
                    d2ndx2 = tao / (2 * (x_range[j] - x_range[j] ** 2) ** (3/2))
                    x_adj = U[i, j + 1] + U[i, j - 1]
                    y_adj = 2 * U[i - 1, j] - (2 * step_size + U[i, j + 1] - U[i, j - 1]) * grad_foil
                    dphidn = (2 * step_size + U[i, j + 1] - U[i, j - 1]) * grad_foil
                    diag_sum = grad_foil * (U[i, j + 2] + U[i, j - 2] - 2 * U[i, j])
                    U[i, j] = U[i, j] + alpha * (((x_adj + ((diag_sum * dndx) / 2) + (y_adj * ((dndx ** 2) + 1)) + ((step_size/2) * (dphidn) * (d2ndx2))) / (2 * (2 + dndx ** 2))) - U[i, j])

                # Points on bottom boundary next to corner
                elif i == m - 1:
                    grad_foil = grad_yb_aerofoil(x_range[j], 0.05)
                    dndx = (tao * (2 * x_range[j] - 1)) / np.sqrt(x_range[j] - x_range[j] ** 2)
                    d2ndx2 = tao / (2 * (x_range[j] - x_range[j] ** 2) ** (3/2))
                    x_adj = U[i, j + 1] + U[i, j - 1]
                    y_adj = 2 * U[i - 1, j] - (2 * step_size + U[i, j + 1] - U[i, j - 1]) * grad_foil
                    dphidn = (2 * step_size + U[i, j + 1] - U[i, j - 1]) * grad_foil
                    diag_sum = 0
                    U[i, j] = U[i, j] + alpha * (((x_adj + ((diag_sum * dndx) / 2) + (y_adj * ((dndx ** 2) + 1)) + ((step_size/2) * (dphidn) * (d2ndx2))) / (2 * (2 + dndx ** 2))) - U[i, j])
                
                # For any other point in grid
                else:
                    dndx = (tao * (2 * x_range[j] - 1)) / np.sqrt(x_range[j] - x_range[j] ** 2)
                    d2ndx2 = tao / (2 * (x_range[j] - x_range[j] ** 2) ** (3/2))
                    x_adj = U[i, j + 1] + U[i, j - 1]
                    y_adj = U[i + 1, j] + U[i - 1, j]
                    dphidn = U[i - 1, j] - U[i + 1, j]
                    diag_sum = U[i + 1, j - 1] + U[i - 1, j + 1] - U[i + 1, j + 1] - U[i - 1, j - 1]
                    U[i, j] = U[i, j] + alpha * (((x_adj + ((diag_sum * dndx) / 2) + (y_adj * ((dndx ** 2) + 1)) + ((step_size/2) * (dphidn) * (d2ndx2))) / (2 * (2 + dndx ** 2))) - U[i, j])
        loops += 1

    return np.flip(U, axis=0), X, N, loops


def time_and_plots(type = 'GS'):
    start = time.time()
    if type == 'GS':
        # q, s, r, u, step_size, type
        U, X, Y, loops = finite_diff_laplace(1, 2, 0.75, 0.2, 0.1, 'GS')
    elif type == 'Jacobi':
        U, X, Y, loops = finite_diff_laplace(1, 2, 0.75, 0.2, 0.1, 'Jacobi')
    elif type == 'SOR':
        # q, s, r, u, step_size, alpha
        U, X, Y, loops = finite_diff_laplace_SOR(1, 4, 0.75, 3.5, 0.05, 1)
    elif type == 'Shifted':
        # r, u, step_size, alpha, tao
        U, X, Y, loops = finite_diff_laplace_shifted_SOR(1, 1, 0.1, 1, 0.05)
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
    colorinterpolation = 100
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

# 28.89, 10088
# time_and_plots('GS')
# 40.44, 14477
# time_and_plots('Jacobi')
# 38.29, 10088
# time_and_plots('SOR')
time_and_plots('Shifted')

