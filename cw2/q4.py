import numpy as np

def twoD_wave_solver(xy_steps, t_steps, c, delta = 0.2):
    x = np.linspace(-2, 2, xy_steps + 1)                  # mesh points in x dir
    y = np.linspace(-2, 2, xy_steps + 1)                  # mesh points in y dir
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    t = np.linspace(0, 20, t_steps + 1)
    dt = t[1] - t[0]                 
    # mesh points in time
    x_mesh = (c * dt/dx) ** 2  
    y_mesh = (c * dt/dy) ** 2   
    print(x_mesh, y_mesh)

    u   = np.zeros((xy_steps + 1, xy_steps + 1))  # solution array
    u_n = [np.zeros_like(u), np.zeros_like(u)]  # t-dt, t-2*dt

    for i in range(0, xy_steps + 1):
        for j in range(0, xy_steps + 1):
            r = (x[i] ** 2 + y[i] ** 2)
            if np.abs(r) <= delta:
                u_n[0][i,j] = np.cos((np.pi * r) / (2 * delta))
            else:
                u_n[0][i,j] = 0

    for t_step in range(0, t_steps + 2):
        for i in range(1, xy_steps):
            for j in range(1, xy_steps):
                u_xx = u_n[0][i - 1, j] - 2 * u_n[0][i, j] + u_n[0][i + 1, j]
                u_yy = u_n[0][i, j - 1] - 2 * u_n[0][i, j] + u_n[0][i, j + 1]
                if t_step == 0:
                    u[i,j] = u_n[0][i,j] + \
                                0.5 * (x_mesh * u_xx + y_mesh * u_yy)
                else:
                    u[i,j] = 2 * u_n[0][i,j] - u_n[1][i,j] + \
                                x_mesh * u_xx + y_mesh * u_yy 

        # Boundary condition 

        u[0, :] = 0
        u[:, 0] = 0
        u[-1, :] = 0
        u[:, -1] = 0

        u_n[1], u_n[0] = u_n[0], u_n[1]
        u_n[0] = u.copy()

    return u

print(twoD_wave_solver(10, 10, 0.1))