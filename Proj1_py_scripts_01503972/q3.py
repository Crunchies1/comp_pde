import numpy as np

def order_diff(h, e, order = 2):
    if not(order == 2 or order == 4 or order == 6):
        print(order, " is not 2 or 4 or 6.")
        return 

    x = np.linspace(0, 1, h + 1)
    dx = x[1] - x[0]

    # Get A
    A = np.zeros((h + 1, h + 1))
    A[0, 0] = -1
    A[h, h] = -1.5

    for i in range(1, h):
        A[i, i - 1] = e + np.sqrt(h/4)
        A[i, i] = -2 - h ** 2
        A[i, i + 1] = e + np.sqrt(h/4)

    # Get b
    b = np.zeros(h + 1)
    for i in range(1, h + 1):
        b[i] = 0
    
    return x, A, b