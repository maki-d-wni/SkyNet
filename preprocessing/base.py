import numpy as np
import pandas as pd

from skynet.base import SkyMLBase


class PreProcessor(SkyMLBase):
    def __init__(self, cdate=True, cwspd=True, smooth=True, edge=True, binary=True, norm=True, **kwargs):
        super().__init__(**kwargs)
        self.cdate = cdate
        self.cwspd = cwspd
        self.smooth = smooth
        self.edge = edge
        self.binary = binary
        self.norm = norm

    def fit(self, X=None, y=None, X_test=None, y_test=None):
        super().preprocessing(X_train=X, y_train=y, X_test=X_test, y_test=y_test,
                              cdate=self.cdate, cwspd=self.cwspd, smooth=self.smooth, edge=self.edge,
                              binary=self.binary, norm=self.norm)

    def predict(self):
        pass

    @staticmethod
    def split(X, y, n_folds=3):
        idx = {int(l): np.where(y.values[:, 0] == l)[0] for l in np.unique(y.values[:, 0])}
        spidx = {}
        for i in range(n_folds):
            spidx[i] = {
                int(l): idx[l][i * int(len(idx[l]) / n_folds):(i + 1) * int(len(idx[l]) / n_folds)]
                for l in idx
            }
            cnc = []
            for k in spidx[i]:
                cnc += list(spidx[i][k])
            spidx[i] = cnc

        spX = {i: X.loc[spidx[i]] for i in range(n_folds)}

        if type(y) != pd.DataFrame:
            y = pd.DataFrame(y)
        spy = {i: y.loc[spidx[i]] for i in range(n_folds)}

        return spX, spy
