import numpy as np
import matplotlib.pyplot as plt

Xs = np.array([0.0339, 0.0423, 0.213, 0.257, 0.273, 0.273, 0.450, 0.503, 0.503, \
0.637, 0.805, 0.904, 0.904, 0.910, 0.910, 1.02, 1.11, 1.11, 1.41, \
1.72, 2.03, 2.02, 2.02, 2.02])

Ys = np.array([-19.3, 30.4, 38.7, 5.52, -33.1, -77.3, 398.0, 406.0, 436.0, 320.0, 373.0, \
93.9, 210.0, 423.0, 594.0, 829.0, 718.0, 561.0, 608.0, 1.04E3, 1.10E3, \
840.0, 801.0, 519.0])

N = 24


def var_sample(Z):
    N = len(Z)
    Zmean = (1 / N) * np.sum(Z)
    return (1 /N) * np.sum((Z - Zmean)**2)


def std_sample(Z):
    N = len(Z)
    Zmean = (1/N) * np.sum(Z)
    return np.sqrt((1/(N-1)) * np.sum((Z - Zmean)**2))


def cov_sample(X, Y):
    N = len(X)
    Xmean = (1 / N) * np.sum(X)
    Ymean = (1 / N) * np.sum(Y)
    return (1/(N-1)) * np.sum((X - Xmean) * (Y - Ymean))


def corr_coefficient(X, Y):
    Sx = std_sample(X)
    Sy = std_sample(Y)
    return cov_sample(X, Y) * 1/Sx * 1/Sy

    # N = len(X)
    # Xmean = (1 / N) * np.sum(X)
    # Ymean = (1 / N) * np.sum(Y)
    # return (1 / (N - 1)) * np.sum((X - Xmean)/Sx * (Y - Ymean)/Sy)


def b_slope(X, Y):
    N = len(X)
    Xmean = (1 / N) * np.sum(X)
    Ymean = (1 / N) * np.sum(Y)
    numerator = np.sum((X - Xmean) * (Y - Ymean))
    denominator = np.sum((X - Xmean)**2)
    return numerator / denominator


def b_not(X, Y):
    N = len(X)
    Xmean = (1 / N) * np.sum(X)
    Ymean = (1 / N) * np.sum(Y)
    return Ymean - (b_slope(X, Y) * Xmean)


def fit_err_func(X, Y):
    b1 = b_slope(X, Y)
    b0 = b_not(X, Y)
    return np.sum(((b1*X) + b0 - Y)**2)


def predictorY(X, Y, x):
    b1 = b_slope(X, Y)
    b0 = b_not(X, Y)
    return (b1*x) + b0


# this function verify the 'Predictive model of distance'
def predictorX(X, Y, y):
    a1 = b_slope(Xs, Ys) * (var_sample(Xs)/var_sample(Ys))
    a0 = b_not(Y, X) # here invertes the order
    return (a1*y) + a0, a1

# coefficient of determination
def coeff_determ(X, Y):
    Sqres = np.sum((Y - predictorX(X, Y, X))**2)
    Ymean = (1 / N) * np.sum(Y)
    Sqdot = np.sum(Y - Ymean)
    return 1 - (Sqres/Sqdot)


def residuas(X, Y):
    Yhat = predictorY(X, Y, X)
    return Y - Yhat


plt.text(min(Xs), max(Ys), 'X mean: %s, Y mean: %s\nX std: %s, Y std: %s'
         % (round(Xs.mean(), 4), round(Ys.mean(), 4), round(std_sample(Xs), 4), round(std_sample(Ys), 4)))
plt.scatter(Xs, Ys, s=10)
plt.plot(Xs, predictorY(Xs, Ys, Xs), 'r-')
plt.xlabel('Distance (Mpc)')
plt.ylabel('Apparent velocuty (km/s)')
plt.show()