from turtle import color
import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt
import pytest
import time
import math
from q1 import forward_euler, error, analytic_soln

def tridiag(A, rhs):
    '''
    Solving Ax = rhs given A and rhs
    (inputs)
    A: n x n numpy matrix
    rhs: n-dimensional numpy vector
    (outputs)
    x: n-dimensional numpy vector
    '''
    _, n = A.shape
    # a, b, c are the relevant diagonals of A
    a = np.zeros(n - 1)
    b = np.zeros(n)
    c = np.zeros(n - 1)
    for i in range(n):
        b[i] = A[i, i]
        if i != (n - 1):
            a[i] = A[i, i + 1]
        if i != 0:
            c[i - 1] = A[i, i - 1]
        
    
    aa, bb, cc, rhs_copy = map(np.array, (a, b, c, rhs)) 

    for j in range(1, n):
        temp = aa[j - 1]/bb[j - 1]
        bb[j] -= temp * cc[j - 1] 
        rhs_copy[j] -= temp * rhs_copy[j - 1]
        	 
    bb[-1] = rhs_copy[-1] / bb[-1]

    for j in range(n - 2, -1, -1):
        bb[j] = (rhs_copy[j] - cc[j] * bb[j + 1]) / bb[j]

    return bb


def implicit(h, k):
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

    # Data structures for the linear system
    A = np.zeros((h + 1, h + 1))
    b = np.zeros(h + 1)

    # Set the A matrix relevant diagonals
   
    for i in range(1, h):
        A[i, i - 1] =  -1 * F
        A[i, i + 1] =  -1 * F
        A[i, i] = 1 + 2 * F 
     
    A[0, 0] = A[h, h] = 1

    # Initial conditions
    for i in range(0, h + 1):
        prev_u[i] = 1
    
    # iterate through time
    for _ in range(0, k):
        # calculate b and solve Ax = b
        for i in range(1, h):
            b[i] = - prev_u[i]
        b[0] = b[h] = 0
        cur_u[:] = tridiag(A, b)

        # Update u 
        prev_u[:] = cur_u

    # Return final u value
    return cur_u 

def crank_nicholson(h, k):
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

    # Data structures for the linear system
    A = np.zeros((h + 1, h + 1))
    A_rhs = np.zeros((h + 1, h + 1))
    b = np.zeros(h + 1)

    # Set the A matrix relevant diagonals
    for i in range(1, h):
        A[i, i - 1] =  -0.5 * F
        A[i, i + 1] =  -0.5 * F
        A[i, i] = 1 + F 
    A[0, 0] = A[h, h] = 1

    # Set the A_rhs matrix relevant diagonals
    for i in range(1, h):
        A_rhs[i, i - 1] =  0.5 * F
        A_rhs[i, i + 1] =  0.5 * F
        A_rhs[i, i] = 1 - F 
    A_rhs[0, 0] = A_rhs[h, h] = 1

    # Initial conditions
    for i in range(0, h + 1):
        prev_u[i] = 1
    
    # iterate through time
    for _ in range(0, k):
        # calculate b and solve Ax = A_rhs @ b
        for i in range(1, h):
            b[i] = - prev_u[i]
        b[0] = b[h] = 0
        cur_u[:] = tridiag(A, A_rhs @ b)

        # Update u 
        prev_u[:] = cur_u

    # Return final u value
    return cur_u 


# Latest recorded times: (i) = 6.23125. (c) = 7.28125.
def time_diffs(n):
    average_times = [0, 0]
    for _ in range(n):
        start_implicit = time.process_time()
        implicit(500, 4000)
        end_implicit = time.process_time()
        implicit_time = end_implicit - start_implicit
        start_crank = time.process_time()
        crank_nicholson(500, 4000)
        end_crank = time.process_time()
        crank_time = end_crank - start_crank
        average_times[0] += implicit_time
        average_times[1] += crank_time
    print(average_times[0] / n, average_times[1] / n)

# Latest recorded times: (e) = 2.135625.
def time_exp(n):
    average_times = 0
    for _ in range(n):
        start_explicit = time.process_time()
        forward_euler(500, 4000)
        end_explicit = time.process_time()
        explicit_time = end_explicit - start_explicit
        average_times += explicit_time
    print(average_times / n)

def plot_error_h(type = 1):
    # fix k as 10000 and vary h 

    if type == 1:
        u_002 = analytic_soln(0.02, 0.075)
        err_arr = []
        err_arr_c = []
        err_arr_i = []
        hs = np.arange(50, 301, 50)
        for h in hs:
            u_num = forward_euler(h, 10000)
            u_num_c = crank_nicholson(h, 10000)
            u_num_i = implicit(h, 10000)
            if math.isnan(u_num[int(h / 50)]):
                err_arr.append(10)
                err_arr_c.append(10)
                err_arr_i.append(10)
            else:
                err_arr.append(error(u_num[int(h / 50)], u_002))
                err_arr_c.append(error(u_num_c[int(h / 50)], u_002))
                err_arr_i.append(error(u_num_i[int(h / 50)], u_002))
            
    
    else:
        u_05 = analytic_soln(0.5, 0.075)
        err_arr = []
        err_arr_c = []
        err_arr_i = []
        hs = np.arange(50, 301, 50)
        for h in hs:
            u_num = forward_euler(h, 10000)
            u_num_c = crank_nicholson(h, 10000)
            u_num_i = implicit(h, 10000)
            if math.isnan(u_num[int(h / 2)]):
                err_arr.append(10)
                err_arr_c.append(10)
                err_arr_i.append(10)
            else:
                err_arr.append(error(u_num[int(h / 2)], u_05))
                err_arr_c.append(error(u_num_c[int(h / 2)], u_05))
                err_arr_i.append(error(u_num_i[int(h / 2)], u_05))
    
    # plotting the points
    plt.plot(hs, err_arr, color="orange")
    plt.plot(hs, err_arr_c, color="crimson")
    plt.plot(hs, err_arr_i, color="teal")
    plt.legend(['FE', 'CN', 'I'])
    
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

def plot_error_k(type = 1):
    if type == 1:
        u_002 = analytic_soln(0.02, 0.075)
        err_arr = []
        err_arr_c = []
        err_arr_i = []
        ks = np.arange(5, 502, 5)
        for k in ks:
            u_num = forward_euler(50, k)
            u_num_c = crank_nicholson(50, k)
            u_num_i = implicit(50, k)
            if math.isnan(u_num[1]):
                err_arr.append(10)
                err_arr_c.append(10)
                err_arr_i.append(10)
            else:
                err_arr.append(error(u_num[1], u_002))
                err_arr_c.append(error(u_num_c[1], u_002))
                err_arr_i.append(error(u_num_i[1], u_002))
            
    
    else:
        u_05 = analytic_soln(0.5, 0.075)
        err_arr = []
        err_arr_c = []
        err_arr_i = []
        ks = np.arange(5, 502, 5)
        for k in ks:
            u_num = forward_euler(50, k)
            u_num_c = crank_nicholson(50, k)
            u_num_i = implicit(50, k)
            if math.isnan(u_num[25]):
                err_arr.append(10)
                err_arr_c.append(10)
                err_arr_i.append(10)
            else:
                err_arr.append(error(u_num[25], u_05))
                err_arr_c.append(error(u_num_c[25], u_05))
                err_arr_i.append(error(u_num_i[25], u_05))
    
    # plotting the points
    plt.plot(ks, err_arr, color="orange")
    plt.plot(ks, err_arr_c, color="crimson")
    plt.plot(ks, err_arr_i, color="teal")
    plt.legend(['FE', 'CN', 'I'])
    
    # naming the x axis
    plt.xlabel('k')
    # naming the y axis
    plt.ylabel('Error of numerical value')
    
    if type == 1:
        plt.title('Error against k with h fixed to 50 at x = 0.02')
    else:
        plt.title('Error against k with h fixed to 50 at x = 0.5')
    
    # function to show the plot
    plt.show()
    return

def plot_error_r():
    u_05 = analytic_soln(0.5, 0.075)
    err_arr = []
    err_arr_c = []
    err_arr_i = []
    hs = np.arange(10, 301, 10)
    rs = []
    for h in hs:
        u_num = forward_euler(h, 10000)
        u_num_c = crank_nicholson(h, 10000)
        u_num_i = implicit(h, 10000)
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
        if math.isnan(u_num[int(h / 2)]):
            err_arr.append(10)
        else:
            err_arr.append(error(u_num[int(h / 2)], u_05))

        if math.isnan(u_num_c[int(h / 2)]):
            err_arr_c.append(10)
        else:
            err_arr_c.append(error(u_num_c[int(h / 2)], u_05))

        if math.isnan(u_num_i[int(h / 2)]):
            err_arr_i.append(10)
        else:
            err_arr_i.append(error(u_num_i[int(h / 2)], u_05))

    # plotting the points
    plt.plot(rs, err_arr, color="orange")
    plt.plot(rs, err_arr_c, color="crimson")
    plt.plot(rs, err_arr_i, color="teal")
    plt.legend(['FE', 'CN', 'I'])
    
    # naming the x axis
    plt.xlabel('r')
    plt.xlim((0, 1))
    # naming the y axis
    plt.ylabel('Error of numerical value')
    # giving a title to my graph
    plt.title('Error from varying r at x = 0.5')
    
    # function to show the plot
    plt.show()
    return

# time_diffs(50)
# time_exp(50)
# plot_error_h(1)
# plot_error_h(2)
# plot_error_k(1)
# plot_error_k(2)
# plot_error_r()

@pytest.mark.parametrize('N', [5, 10])
def test_tri_diag(N):

    A = np.zeros((N, N))
    A[0, 0] = 2.0
    A[0, 1] = -1.0

    RHS = np.zeros(N)
    RHS[0] = 1.0

    for i in range(1, N - 1):
        A[i, i - 1] = -1.0
        A[i, i] = 2.0
        A[i, i + 1] = -1.0
        RHS[i] = 1.0

    A[N - 1, N - 1] = 2.0
    A[N - 1, N - 2] = -1.0
    RHS[N - 1] = 1.0

    u = tridiag(A,RHS)
    true_u = scipy.linalg.solve(A, RHS)

    assert(np.allclose(true_u, u))
  