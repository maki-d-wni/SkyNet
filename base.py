import numpy as np
import pandas as pd

from abc import ABCMeta, abstractclassmethod

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score

from skynet.data_handling import get_init_vis_level
from skynet.data_handling import concat
from skynet.feature_engineering import convert_date
from skynet.feature_engineering import convert_wind_speed
from skynet.feature_engineering import conv_pressure_surface
from skynet.evaluation import rmse
from skynet.evaluation import confusion_matrix


class SkyMLBase(metaclass=ABCMeta):
    def __init__(self, **kwargs):
        pass

    @abstractclassmethod
    def fit(self):
        pass

    @abstractclassmethod
    def predict(self):
        pass

    def preprocessing(self, X_train=None, y_train=None, X_test=None, y_test=None,
                      cdate=True, cwspd=True, smooth=True, edge=True,
                      binary=True, norm=True):
        if X_train is not None:
            X_train, y_train = self.__preprocessing(
                X_train, y_train, binary, norm, cdate, cwspd, smooth, edge
            )
            self.__train_data(X_train, y_train)

        if X_test is not None:
            X_test, y_test = self.__preprocessing(
                X_test, y_test, binary, norm, cdate, cwspd, smooth, edge
            )
            self.__test_data(X_test, y_test)

        return X_train, y_train, X_test, y_test

    def reset(self, **kwargs):
        self.__init__(**kwargs)

    def __train_data(self, X, y):
        self.X_train = X
        self.y_train = y

    def __test_data(self, X, y):
        self.X_test = X
        self.y_test = y

    def __preprocessing(self, X, y, binary=False, norm=False, cdate=True, cwspd=True, smooth=True, edge=True):
        if cdate and "date" in X.keys():
            X = convert_date(X)

        if cwspd:
            X = convert_wind_speed(X)

        if smooth:
            physical_quantity = ["Vert", "Temp", "Rela"]
            kernels = {"smooth_ks%d" % ks: np.ones((1, ks)) / ks for ks in [3, 5, 7, 9]}
            for f in physical_quantity:
                for k in kernels:
                    h_smooth = conv_pressure_surface(X, kernel=kernels[k], kernel_key=k,
                                                     physical_quantity=f, mode="valid")
                    X = concat(X, h_smooth)

        if edge:
            physical_quantity = ["Vert", "Temp", "Rela"]
            kernels = {"edge": np.array([[-1 / 2, 0, 1 / 2]])}
            for f in physical_quantity:
                for k in kernels:
                    h_edge = conv_pressure_surface(X, kernel=kernels[k], kernel_key=k,
                                                   physical_quantity=f, mode="valid")
                    X = concat(X, h_edge)

        self.feature_ = list(X.keys())

        if binary:
            self.target_ = ["binary"]
            threshold = int(len(get_init_vis_level()) / 2)
            y = np.where(y > threshold, 0, 1)

        if norm:
            ss = StandardScaler()
            X = pd.DataFrame(ss.fit_transform(X), columns=X.keys())

        return X, y

    def evaluate(self, p, y, threshold=None, conf_mat=False, eval_index=False):
        label = np.array(list(get_init_vis_level().keys()))
        if label.min() != 0. or label.max() != 1.:
            if threshold is None:
                threshold = int(len(label) / 2)

            p = np.where(p > threshold, 0, 1)
            y = np.where(y > threshold, 0, 1)

        pt00 = np.where((p == 0.) & (y == 0.))[0]
        pt11 = np.where((p == 1.) & (y == 1.))[0]
        pt01 = np.where((p == 0.) & (y == 1.))[0]
        pt10 = np.where((p == 1.) & (y == 0.))[0]

        if conf_mat:
            confusion_matrix(len(pt11), len(pt00), len(pt10), len(pt01))

        self.__accuracy(accuracy_score(y, p))
        self.__threat_score(len(pt11) / (len(pt10) + len(pt11) + len(pt01)))
        self.__rmse(rmse(y, p))

        if eval_index:
            print(
                "Accuracy     : {:.3}\n"
                "Thread Score : {:.3}\n"
                "RMSE         : {:.3}\n".format(self.accuracy_, self.threat_score_, self.rmse_)
            )

        print(classification_report(y_true=y, y_pred=p))

        return self.threat_score_, self.accuracy_, self.rmse_

    def __accuracy(self, val):
        self.accuracy_ = val

    def __threat_score(self, val):
        self.threat_score_ = val

    def __rmse(self, val):
        self.rmse_ = val


def fit(clf, X, y, sample_weight=None, **kwargs):
    if X is not None and y is not None:
        clf.fit(X, y, sample_weight, **kwargs)
    else:
        raise Exception("You need preprocessing.")


def predict(clf, X):
    if X is not None:
        return clf.predict(X)
    else:
        raise Exception("You need preprocessing.")
