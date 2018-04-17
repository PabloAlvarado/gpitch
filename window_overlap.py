import numpy as np
from scipy import signal


def windowed(x, y, ws=8001):
    n = x.size
    l = (ws-1)/2
    xout = []
    yout = []
    nw = (n - ws)/l + 1
    for i in range(nw):
        if i == 0 :
            win = signal.hann(ws).reshape(-1, 1)
            win[0:l] = 1.
        elif i == nw-1:
            win = signal.hann(ws).reshape(-1, 1)
            win[-l:] = 1.
        else:
            win = signal.hann(ws).reshape(-1, 1)
        xout.append(x[i*l : i*l + ws].copy().reshape(-1, 1))
        yout.append(y[i*l : i*l + ws].copy().reshape(-1, 1)*win)
        
    return xout, yout


def merged_y(y, ws=8001):
    l = (ws-1)/2
    nw = len(y)
    n = (ws-1)/2 * (nw - 1) + ws
    yout = np.zeros((n, 1))
    
    yout[0:l] = y[0][0:l]
    yout[-l-1:] = y[-1][-l-1:]
    
    for i in range(nw-1): 
        yout[(i+1)*l : (i+2)*l] = y[i][-l-1:-1].copy() + y[i+1][0:l].copy()
    return yout


def merged_x(x, ws=8001):
    l = (ws-1)/2
    nw = len(x)
    n = (ws-1)/2 * (nw - 1) + ws
    xout = np.zeros((n, 1))
    
    xout[0:l] = x[0][0:l]
    xout[-l-1:] = x[-1][-l-1:]
    
    for i in range(nw-1): 
        xout[(i+1)*l : (i+2)*l] = x[i][-l-1:-1].copy()
    
    return xout