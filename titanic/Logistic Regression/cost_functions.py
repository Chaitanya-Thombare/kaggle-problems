import numpy as np

def LogLoss(Yp, Yg):
    cost = -(Yg * np.log(Yp) + (1 - Yg) * np.log(1 - Yp))
    return np.mean(cost)