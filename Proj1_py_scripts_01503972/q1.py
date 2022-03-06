import numpy as np
import matplotlib.pyplot as plt
import math

# error equation
def error(u_num, u_analytic):
    if u_analytic == 0:
        return 1000
    return np.log10(1 + np.abs(u_num - u_analytic))

def analytic_soln(x, t, maxit = 10000, threshold = 0.000001):
    # Initialize u value
    u = 0

    # Set cut off point 
    for i in range(maxit):
        # Calculate exponential term using analytic equation
        e_term = np.exp(-1 * ((2 * i + 1) ** 2) * (np.pi ** 2) * t)
        # Calculate sin term using analytic equation
        sin_term = np.sin((2 * i + 1) * np.pi * x)
        # Calculate current sum term
        total = (1 / ((2 * i) + 1)) * e_term * sin_term

        # Increment total and check for break condition
        u += total
        if total < threshold:
            break

    # u is analytic solution at point x and t    
    u = u * (4 / np.pi)
    return u

def forward_euler(h, k):
    # discrete points in space
    x = np.linspace(0, 1, h + 1) 
    # distance of step
    dx = x[1] - x[0]
    # discrete points in time
    t = np.linspace(0, 0.075, k)
    # time steps 
    dt = t[1] - t[0]

    # define u and previous u
    cur_u   = np.zeros(h + 1)
    prev_u = np.zeros(h + 1)

    # Define mesh fourier number
    F = dt / (dx ** 2)

    # Initial conditions 
    for i in range(0, h + 1):
        prev_u[i] = 1

    # Iterate through time steps
    for _ in range(0, k):
        # equation for computing u
        for i in range(1, h):
            cur_u[i] = prev_u[i] + F * (prev_u[i-1] - 2 * prev_u[i] + prev_u[i+1])
        # Boundary conditions
        cur_u[0] = 0
        cur_u[h] = 0

        # Swap variables
        prev_u, cur_u = cur_u, prev_u

    # Return final u value
    return prev_u

def plot_error_k(type = 1):
    # fix h as 50 and vary k 
    if type == 1:
        u_002 = analytic_soln(0.02, 0.075)
        err_arr = []
        ks = np.arange(5, 502, 5)
        for k in ks:
            u_num = forward_euler(50, k)
            err_arr.append(error(u_num[1], u_002))
    
    else:
        u_05 = analytic_soln(0.5, 0.075)
        err_arr = []
        ks = np.arange(5, 502, 5)
        for k in ks:
            u_num = forward_euler(50, k)
            err_arr.append(error(u_num[25], u_05))
    
    # plotting the points
    plt.plot(ks, err_arr)
    
    # naming the x axis
    plt.xlabel('k')
    # naming the y axis
    plt.ylabel('Error of numerical value')
    
    # giving a title to my graph
    if type == 1:
        plt.title('Error against k with h fixed to 50 at x = 0.02')
    else:
        plt.title('Error against k with h fixed to 50 at x = 0.5')
    
    # function to show the plot
    plt.show()
    return


def plot_error_h(type = 1):
    # fix k as 10000 and vary h 

    if type == 1:
        u_002 = analytic_soln(0.02, 0.075)
        err_arr = []
        hs = np.arange(50, 502, 50)
        for h in hs:
            u_num = forward_euler(h, 10000)
            if math.isnan(u_num[int(h / 50)]):
                err_arr.append(10)
            else:
                err_arr.append(error(u_num[int(h / 50)], u_002))
    
    else:
        u_05 = analytic_soln(0.5, 0.075)
        err_arr = []
        hs = np.arange(50, 502, 50)
        for h in hs:
            u_num = forward_euler(h, 10000)
            if math.isnan(u_num[int(h / 2)]):
                err_arr.append(10)
            else:
                err_arr.append(error(u_num[int(h / 2)], u_05))
    
    # plotting the points
    plt.plot(hs, err_arr)
    
    # naming the x axis
    plt.xlabel('h')
    plt.xlim((50, 300))
    # naming the y axis
    plt.ylabel('Error of numerical value')
    plt.ylim((-0.00001, 0.00003))
    # giving a title to my graph
    if type == 1:
        plt.title('Error against h with k fixed at 10000 at x = 0.02')
    else:
        plt.title('Error against h with k fixed at 10000 at x = 0.5')
    
    # function to show the plot
    plt.show()
    return

def plot_error_r(type = 1):
    # fix k as 10000 and vary h 

    if type == 1:
        u_002 = analytic_soln(0.02, 0.075)
        err_arr = []
        hs = np.arange(50, 502, 50)
        rs = []
        for h in hs:
            u_num = forward_euler(h, 10000)
            if math.isnan(u_num[int(h / 50)]):
                x = np.linspace(0, 1, h + 1) 
                # distance of step
                dx = x[1] - x[0]
                # discrete points in time
                t = np.linspace(0, 0.075, 10000)
                # time steps 
                dt = t[1] - t[0]
                # Define mesh fourier number
                F = dt / (dx ** 2)
                rs.append(F)
                err_arr.append(10)
            else:
                x = np.linspace(0, 1, h + 1) 
                # distance of step
                dx = x[1] - x[0]
                # discrete points in time
                t = np.linspace(0, 0.075, 10000)
                # time steps 
                dt = t[1] - t[0]
                # Define mesh fourier number
                F = dt / (dx ** 2)
                rs.append(F)
                err_arr.append(error(u_num[int(h / 50)], u_002))
    
    else:
        u_05 = analytic_soln(0.5, 0.075)
        err_arr = []
        hs = np.arange(10, 301, 10)
        rs = []
        for h in hs:
            u_num = forward_euler(h, 10000)
            if math.isnan(u_num[int(h / 2)]):
                x = np.linspace(0, 1, h + 1) 
                # distance of step
                dx = x[1] - x[0]
                # discrete points in time
                t = np.linspace(0, 0.075, 10000)
                # time steps 
                dt = t[1] - t[0]
                # Define mesh fourier number
                F = dt / (dx ** 2)
                rs.append(F)
                err_arr.append(10)
            else:
                x = np.linspace(0, 1, h + 1) 
                # distance of step
                dx = x[1] - x[0]
                # discrete points in time
                t = np.linspace(0, 0.075, 10000)
                # time steps 
                dt = t[1] - t[0]
                # Define mesh fourier number
                F = dt / (dx ** 2)
                rs.append(F)
                err_arr.append(error(u_num[int(h / 2)], u_05))
    
    # plotting the points
    plt.plot(rs, err_arr)
    
    # naming the x axis
    plt.xlabel('r')
    plt.xlim((0, 1))
    # naming the y axis
    plt.ylabel('Error of numerical value')
    plt.ylim((-0.00001, 0.0001))
    # giving a title to my graph
    if type == 1:
        plt.title('Error from varying r at x = 0.02')
    else:
        plt.title('Error from varying r at x = 0.5')
    
    # function to show the plot
    plt.show()
    return

# plot_error_k(1)
# plot_error_k(2)
# plot_error_h(1)
# plot_error_h(2)
# plot_error_r(1)
# plot_error_r(2)