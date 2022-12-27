import numpy as np


def ema(data, r=0.7):
    ret = []
    for x in data:
        if len(ret) == 0:
            ret.append(x)
        else:
            ret.append(ret[-1] * (1 - r) + x * r)
    return np.array(ret)
