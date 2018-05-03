import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from gpitch.methods import logistic


def windowed(x, y, ws):
    n = x.size
    l = (ws-1)/2
    xout = []
    yout = []
    nw = (n - ws)/l + 1
    for i in range(nw):
        # if i == 0 :
        #     win = signal.hann(ws).reshape(-1, 1)
        #     win[0:l] = 1.
        # elif i == nw-1:
        #     win = signal.hann(ws).reshape(-1, 1)
        #     win[-l:] = 1.
        # else:
        #     win = signal.hann(ws).reshape(-1, 1)
        xout.append(x[i*l : i*l + ws].copy().reshape(-1, 1))
        yout.append(y[i*l : i*l + ws].copy().reshape(-1, 1))

    return xout, yout


def merged_y(y, ws):
    l = (ws-1)/2
    nw = len(y)
    n = (ws-1)/2 * (nw - 1) + ws

    # for i in range(len(y)):
    #     if i == 0 :
    #         win = signal.hann(ws).reshape(-1, 1)
    #         win[0:l] = 1.
    #     elif i == nw-1:
    #         win = signal.hann(ws).reshape(-1, 1)
    #         win[-l:] = 1.
    #     else:
    #         win = signal.hann(ws).reshape(-1, 1)
    #     y[i] = y[i]/(win + 0.0001)

    for i in range(len(y)):
        if i == 0 :
            win = signal.hann(ws).reshape(-1, 1)
            win[0:l] = 1.
        elif i == nw-1:
            win = signal.hann(ws).reshape(-1, 1)
            win[-l:] = 1.
        else:
            win = signal.hann(ws).reshape(-1, 1)
        y[i] = y[i]*win

    yout = np.zeros((n, 1))
    yout[0:l] = y[0][0:l]
    yout[-l-1:] = y[-1][-l-1:]

    for i in range(nw-1):
        yout[(i+1)*l : (i+2)*l] = y[i][-l-1:-1].copy() + y[i+1][0:l].copy()
    return yout


def merged_x(x, ws):
    l = (ws-1)/2
    nw = len(x)
    n = (ws-1)/2 * (nw - 1) + ws
    xout = np.zeros((n, 1))

    xout[0:l] = x[0][0:l]
    xout[-l-1:] = x[-1][-l-1:]

    for i in range(nw-1):
        xout[(i+1)*l : (i+2)*l] = x[i][-l-1:-1].copy()

    return xout


def merge_all(inlist):
    outlist = [ [[], [], []],
                [[], [], []],
                [[], [], []],
                [[], [], []],
                [[], [], []] ]
    nrow = len(outlist)
    ncol = len(inlist)
    
    for j in range(nrow):
        for i in range(ncol):        
            outlist[j][0].append(inlist[i][j][0])
            outlist[j][1].append(inlist[i][j][1])
            outlist[j][2].append(inlist[i][j][2])
    
    return outlist


def append_sources(rmerged):
    s1_l = []
    s2_l = []
    s3_l = []
    for i in range(len(rmerged[0][0])):
        s1_l.append( logistic(rmerged[0][0][i].copy()) * rmerged[2][0][i].copy() )
        s2_l.append( logistic(rmerged[0][1][i].copy()) * rmerged[2][1][i].copy() )
        s3_l.append( logistic(rmerged[0][2][i].copy()) * rmerged[2][2][i].copy() )
    return s1_l, s2_l, s3_l


def get_results_arrays(x, y, sl, ws):
    s1 = merged_y(sl[0], ws)
    s2 = merged_y(sl[1], ws)
    s3 = merged_y(sl[2], ws)

    x = merged_x(x, ws)
    y = merged_y(y, ws)

    s1_trim = s1[0:-1].reshape(-1, 1)
    s2_trim = s2[0:-1].reshape(-1, 1)
    s3_trim = s3[0:-1].reshape(-1, 1)

    x_trim = x[0:-1].reshape(-1, 1)
    y_trim = y[0:-1].reshape(-1, 1)

    s = [s1_trim, s2_trim, s3_trim]
    return x_trim, y_trim, s


def plot_sources(x, y, s):
    plt.figure(figsize=(16, 9))

    plt.subplot(3,1,1)
    plt.plot(x, y)
    plt.plot(x, s[0])
    plt.ylim(-1, 1)

    plt.subplot(3,1,2)
    plt.plot(x, y)
    plt.plot(x, s[1])
    plt.ylim(-1, 1)

    plt.subplot(3,1,3)
    plt.plot(x, y)
    plt.plot(x, s[2])
    plt.ylim(-1, 1)


def plot_patches(x, y, rm, s1_l, s2_l, s3_l):
    num_patches = len(rm[0][0])
    for i in range(num_patches):
        plt.figure(1000, figsize=(16, 4))
        plt.plot(x[i], i + s1_l[i], 'C0')
        plt.plot(x[i], i + y[i], 'C1')


        plt.figure(2000, figsize=(16, 4))
        plt.plot(x[i], i + s2_l[i], 'C0')
        plt.plot(x[i], i + y[i], 'C1')


        plt.figure(3000, figsize=(16, 4))
        plt.plot(x[i], i + s3_l[i], 'C0')
        plt.plot(x[i], i + y[i], 'C1')
