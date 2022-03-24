import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def cx(x):
    return (16 - 3 * (x ** 2)) / 4

def one_dimensional_wave_solver(x_steps, t_steps, delta, boundary = 'pass'):
    # discrete points in space
    x = np.linspace(-2, 2, x_steps + 1) 
    # distance of step
    dx = x[1] - x[0]
    # discrete points in time
    t = np.linspace(0, 20, t_steps + 1)
    # time steps 
    dt = t[1] - t[0]

    # define cur_u and previous u
    cur_u = np.zeros(x_steps + 1)
    prev_u = np.zeros((x_steps + 1, t_steps + 2))

    if boundary == 'pass':
        print("pass")
    else:
        print(boundary)

    # Define mesh constant (dt)**2/(dx)**2
    c = 1
    mesh = (c * dt) / dx
    print(mesh)

    # Initial conditions 
    for i in range(0, x_steps + 1):
        if (-delta <= x[i] and x[i] <= delta):
            prev_u[i, 0] = np.cos((np.pi * x[i]) / (2 * delta))
        else:
            prev_u[i, 0] = 0

    # Calculate cur_u for t = 0
    for i in range(1, x_steps):
        cur_u[i] = prev_u[i, 0] + (cx(x[i]) * (mesh ** 2) / 2) * (prev_u[i - 1, 0] - 2 * prev_u[i, 0] + prev_u[i + 1, 0])
    # Boundary conditions
    cur_u[0] = prev_u[0, 0] - (cx(x[0]) * (mesh ** 2)) * (prev_u[1, 0] - prev_u[0, 0])
    cur_u[-1] = prev_u[-1, 0] + (cx(x[-1]) * (mesh ** 2)) * (prev_u[-1, 0] - prev_u[-2, 0])

    # Swap variables
    prev_u[:, 1] = cur_u.copy()

    # Iterate through time steps
    for t in range(0, t_steps + 1):
        # For the first time step
        if t == 0:
            for i in range(1, x_steps):
                cur_u[i] = prev_u[i, t] + (cx(x[i]) * (mesh ** 2) / 2) * (prev_u[i - 1, t] - 2 * prev_u[i, t] + prev_u[i + 1, t])
            # Boundary conditions
            cur_u[0] = prev_u[0, 0] - (cx(x[0]) * (mesh ** 2)) * (prev_u[1, 0] - prev_u[0, 0])
            cur_u[-1] = prev_u[-1, 0] + (cx(x[-1]) * (mesh ** 2)) * (prev_u[-1, 0] - prev_u[-2, 0])

        else:
            # equation for computing u
            for i in range(1, x_steps):
                cur_u[i] = 2 * prev_u[i, t] - prev_u[i, t - 1] + cx(x[i]) * (mesh ** 2) * (prev_u[i - 1, t] - 2 * prev_u[i, t] + prev_u[i + 1, t])
            # Boundary conditions
            if boundary == 'pass':
                '''
                cur_u[0] = prev_u[0, t] + cx(x[0]) * mesh * (prev_u[1, t] - prev_u[0, t])
                cur_u[-1] = prev_u[-1, t] - cx(x[-1]) * mesh * (prev_u[-2, t] - prev_u[-1, t])
                '''
                cur_u[0] = (2 * prev_u[0, t] - (mesh - 1) * prev_u[0, t - 1] + 2 * (mesh ** 2) * (prev_u[1, t] - prev_u[0, t])) / (mesh + 1)
                cur_u[-1] = (2 * prev_u[-1, t] - (mesh - 1) * prev_u[-1, t - 1] + 2 * (mesh ** 2) * (prev_u[-2, t] - prev_u[-1, t])) / (mesh + 1)
            else:
                cur_u[0] = 2 * prev_u[0, t] - prev_u[0, t - 1] + 2 * (mesh ** 2) * (prev_u[1, t] - prev_u[0, t])
                cur_u[-1] = 2 * prev_u[-1, t] - prev_u[-1, t - 1] + 2  * (mesh ** 2) * (prev_u[-2, t] - prev_u[-1, t])

        # Swap variables
        prev_u[:, t + 1] = cur_u.copy()

    # Plot U
    '''
    plt.plot(np.arange(0, t_steps + 2), prev_u[int(x_steps / 2)], 'go--', linewidth=1, markersize=2)
    plt.xlabel("time step")
    plt.ylabel("U")
    if boundary == 'pass':
        plt.title("U against time at x = 0 with open boundaries")
    else:
        plt.title("U against time at x = 0 with solid wall boundaries")
    plt.show()

    plt.plot(np.arange(0, t_steps + 2), prev_u[0], 'go--', linewidth=1, markersize=2)
    plt.xlabel("time step")
    plt.ylabel("U")
    if boundary == 'pass':
        plt.title("U against time at x = -2 with open boundaries")
    else:
        plt.title("U against time at x = -2 with solid wall boundaries")
    plt.show()

    plt.plot(np.arange(0, t_steps + 2), prev_u[-1], 'go--', linewidth=1, markersize=2)
    plt.xlabel("time step")
    plt.ylabel("U")
    if boundary == 'pass':
        plt.title("U against time at x = 2 with open boundaries")
    else:
        plt.title("U against time at x = 2 with solid wall boundaries")
    plt.show()
    '''

    # Return final u value
    return prev_u[:, -1], prev_u

last_val, vals = one_dimensional_wave_solver(40, 1600, 0.1, 'pass')
fig = plt.figure()
ims = []
for i in range(len(vals[0])):
    im = plt.imshow(vals[:, i].reshape(1, -1), animated=True)
    ims.append([im])
ani = animation.ArtistAnimation(fig, ims, interval=50, blit=False,
                                repeat_delay=1000)
plt.show()
# print(one_dimensional_wave_solver_infinite(20, 0.025, 0.01, 'pass'))
