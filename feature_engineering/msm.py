import numpy as np
import pandas as pd
from scipy import signal

from skynet.data_handling import match_keys


def convert_wind_speed(data):
    uco = match_keys(list(data.keys()), "u-co")
    vco = match_keys(list(data.keys()), "v-co")

    for uh, vh in zip(uco, vco):
        u = data[uh]
        v = data[vh]
        wspd = np.sqrt(u ** 2 + v ** 2)
        if uh == "u-component of wind":
            wh = "wind speed"
            data.insert(loc=len(data.keys()), column=wh, value=wspd)
        else:
            wh = "wspd" + uh[4:]
            data.insert(loc=len(data.keys()), column=wh, value=wspd)

    return data


def max_pressure_surface(data, physical_quantity):
    features = list(data.keys())
    exts = match_keys(features, physical_quantity)
    return pd.DataFrame(data[exts].abs().max(axis=1).values, columns=["max_%s" % physical_quantity])


def grad_pressure_surface(data, physical_quantity):
    features = list(data.keys())
    exts = match_keys(features, physical_quantity)
    new_features = ["h_grad_%s" % f for f in exts]
    grad = pd.DataFrame(data[exts].diff(periods=1, axis=1).values, columns=new_features)
    grad.dropna(axis=1, inplace=True)
    return grad


def conv_pressure_surface(data, kernel, kernel_key, physical_quantity, mode="valid"):
    features = list(data.keys())
    exts = match_keys(features, physical_quantity)
    conv = signal.convolve2d(data[exts].values, kernel, mode=mode)
    new_features = ["%s_%s_%d" % (kernel_key, physical_quantity, i) for i in range(conv.shape[1])]
    return pd.DataFrame(conv, columns=new_features)
