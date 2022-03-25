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
    t = np.linspace(0, 10, t_steps + 1)
    # time steps 
    dt = t[1] - t[0]

    # define cur_u and previous u
    cur_u = np.zeros(x_steps + 1)
    prev_u = np.zeros((x_steps + 1, t_steps + 2))

    if boundary == 'pass':
        print("pass")
    else:
        print(boundary)

    # Define mesh constant
    c = 1
    mesh = (c * dt) / dx
    print(dx, dt)
    print(mesh)

    # Initial conditions 
    for i in range(0, x_steps + 1):
        if (-delta <= x[i] and x[i] <= delta):
            prev_u[i, 0] = np.cos((np.pi * x[i]) / (2 * delta))
        else:
            prev_u[i, 0] = 0

    # Iterate through time steps
    for t in range(0, t_steps + 1):
        # For the first time step
        if t == 0:
            for i in range(1, x_steps):
                cur_u[i] = prev_u[i, t] + (cx(x[i]) * (mesh ** 2) / 2) * (prev_u[i - 1, t] - 2 * prev_u[i, t] + prev_u[i + 1, t])
            # Boundary conditions
            if boundary == 'pass':
                
                cur_u[0] = prev_u[0, t] +  mesh * (prev_u[1, t] - prev_u[0, t])
                cur_u[-1] = prev_u[-1, t] - mesh * (prev_u[-1, t] - prev_u[-2, t])
                # cur_u[0] = (2 * prev_u[0, t] - (mesh - 1) * prev_u[0, t - 1] + 2 * (mesh ** 2) * (prev_u[1, t] - prev_u[0, t])) / (mesh + 1)
                # cur_u[-1] = (2 * prev_u[-1, t] - (mesh - 1) * prev_u[-1, t - 1] + 2 * (mesh ** 2) * (prev_u[-2, t] - prev_u[-1, t])) / (mesh + 1)
            else:
                cur_u[0] = 2 * prev_u[0, t] - prev_u[0, t - 1] + 2 * (mesh ** 2) * (prev_u[1, t] - prev_u[0, t])
                cur_u[-1] = 2 * prev_u[-1, t] - prev_u[-1, t - 1] + 2  * (mesh ** 2) * (prev_u[-2, t] - prev_u[-1, t])

        else:
            # equation for computing u
            for i in range(1, x_steps):
                cur_u[i] = 2 * prev_u[i, t] - prev_u[i, t - 1] + cx(x[i]) * (mesh ** 2) * (prev_u[i - 1, t] - 2 * prev_u[i, t] + prev_u[i + 1, t])
            # Boundary conditions
            if boundary == 'pass':
                
                #cur_u[0] = prev_u[0, t] +  mesh * (prev_u[1, t] - prev_u[0, t])
                #cur_u[-1] = prev_u[-1, t] - mesh * (prev_u[-1, t] - prev_u[-2, t])
                # print((2 * prev_u[0, t] - (-mesh - 1) * prev_u[0, t - 1] + (2 * mesh ** 2) * (prev_u[1, t] - prev_u[0, t])) / (-mesh + 1))
                cur_u[0] = (2 * prev_u[0, t] + (mesh - 1) * prev_u[0, t - 1] + (2 * mesh ** 2) * (prev_u[1, t] - prev_u[0, t])) / (mesh + 1)
                cur_u[-1] = (2 * prev_u[-1, t] + (mesh - 1) * prev_u[-1, t - 1] + 2 * (mesh ** 2) * (prev_u[-2, t] - prev_u[-1, t])) / (mesh + 1)
            elif boundary == 'pml':

                cur_u[0] = prev_u[0, t] + mesh * (1 / (1 + 0.3j)) * (prev_u[1, t] - prev_u[0, t]) 
                cur_u[-1] = prev_u[-1, t] - mesh * (1 / (1 + 0.3j)) * (prev_u[-1, t] - prev_u[-2, t]) 

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

last_val, vals = one_dimensional_wave_solver(80, 400, 0.1, 'pass')

fig = plt.figure()
ax = plt.gca()
ax.set_ylim([-0.5, 1])
plts = []             # get ready to populate this list the Line artists to be plotted

for i in range(len(vals[0])):
    p, = plt.plot(vals[:, i], 'k')   # this is how you'd plot a single line...
    plts.append( [p] )           # ... but save the line artist for the animation
ani = animation.ArtistAnimation(fig, plts, interval=50, repeat_delay=3000)   # run the animation
# ani.save('wave.')    # optionally save it to a file

plt.show()
# print(one_dimensional_wave_solver_infinite(20, 0.025, 0.01, 'pass'))
