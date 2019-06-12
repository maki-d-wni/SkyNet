import numpy as np
import pandas as pd


def pearson_correlation(X, y, depth=None):
    fets = np.array(X.keys())
    Xy = pd.concat([X, y], axis=1)
    corr = np.absolute(np.corrcoef(Xy.T)[-1, :-1])

    idx = np.argsort(corr)[::-1]
    if depth is None:
        fets = list(fets[idx])
    else:
        fets = fets[idx][:depth]

    return list(fets)


def relief(X, y, depth=None):
    if "date" in X.keys():
        X = X.drop("date", axis=1)

    if type(y) == pd.DataFrame:
        y = y.values[:, 0]

    if y.min() != 0. or y.max() != 1.:
        label = get_init_vis_level()
        threshold = int(len(label) / 2)

        y = np.where(y > threshold, 0, 1)

    fets = np.array(X.keys())
    m = np.zeros(len(fets))
    for idx in range(len(X)):
        r = np.sqrt(((X.values - X.iloc[idx:idx + 1].values) ** 2).sum(axis=1))

        r[idx] = 10000

        xh = X[y == y[idx]]
        xm = X[y != y[idx]]

        rh = r[y == y[idx]]
        rm = r[y != y[idx]]

        ih = rh.argmin()
        im = rm.argmin()

        near_h = xh.iloc[ih]
        near_m = xm.iloc[im]

        diff_h = np.absolute((X.iloc[idx].values - near_h.values))
        diff_m = np.absolute((X.iloc[idx].values - near_m.values))

        m += diff_m / len(X) - diff_h / len(X)

    if depth is None:
        exfs = fets[m.argsort()[::-1]]
    else:
        exfs = fets[m.argsort()[::-1]][:depth]
    return exfs
