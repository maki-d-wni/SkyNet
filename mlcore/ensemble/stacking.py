import numpy as np
import pandas as pd
from skynet.mlcore.ensemble import forest
from skynet.mlcore.ensemble import RandomForestClassifier
from skynet.mlcore.linear_model import LogisticRegression


class SkyStacking(forest.ForestClassifier):
    def __init__(self, classifiers, meta_classifier, n_folds=3):
        self.classifiers = classifiers
        self.meta_classifier = meta_classifier
        self.n_folds = n_folds

    def fit(self, X=None, y=None):
        shuffled = np.random.permutation(len(X))
        X = X.loc[shuffled]
        y = y.loc[shuffled]

        self.__blend(X, y)

    def predict(self, X=None):
        blend_test = np.zeros((len(X), len(self.classifiers)))

        for n_clf in range(len(self.classifiers)):
            blend_test_ele = np.zeros((len(X), self.n_folds))
            for i in range(self.n_folds):
                blend_test_ele[:, i] = self.classifiers[n_clf].predict(X)

            blend_test[:, n_clf] = blend_test.mean(axis=1)

        return self.meta_classifier.predict(blend_test)

    def __split(self, X, y, cv=None):
        if cv is None:
            cv = self.n_folds

        idx = {int(l): np.where(y.values[:, 0] == l)[0] for l in np.unique(y.values[:, 0])}
        spidx = {}
        for i in range(cv):
            spidx[i] = {
                int(l): idx[l][i * int(len(idx[l]) / cv):(i + 1) * int(len(idx[l]) / cv)]
                for l in idx
            }
            cnc = []
            for k in spidx[i]:
                cnc += list(spidx[i][k])
            spidx[i] = cnc

        spX = {i: X.loc[spidx[i]] for i in range(cv)}

        if type(y) != pd.DataFrame:
            y = pd.DataFrame(y)
        spy = {i: y.loc[spidx[i]] for i in range(cv)}

        return spX, spy

    def __blend(self, X, y):
        spX, spy = self.__split(X, y, self.n_folds)

        blend_train = np.zeros((len(X), len(self.classifiers)))

        for n_clf in range(len(self.classifiers)):
            idx = 0
            for i in range(self.n_folds):
                X_train = pd.concat([spX[n] for n in spX if n != i])
                y_train = pd.concat([spy[n] for n in spy if n != i])
                X_train, y_train = balanced(X_train, y_train)
                X_train = X_train.values
                y_train = y_train.values[:, 0]

                X_valid = spX[i]
                X_valid = X_valid.values

                self.classifiers[n_clf].fit(X_train, y_train)
                blend_train[idx:idx + len(X_valid), n_clf] = self.classifiers[n_clf].predict(X_valid)

                idx += len(X_valid)

        self.meta_classifier.fit(blend_train, y.values[:, 0])


def main():
    import skynet.nwp2d as npd
    import skynet.datasets as skyds
    from skynet import DATA_DIR
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import f1_score

    icao = "RJAA"

    train_data_dir = '%s/MSM/airport.process' % DATA_DIR
    test_data_dir = '%s/skynet' % DATA_DIR

    train = skyds.read_csv('%s/%s.csv' % (train_data_dir, icao))
    test = skyds.read_pkl('%s/test_%s.pkl' % (test_data_dir, icao))

    test['date'] = test['date'].astype(int).astype(str)
    test = npd.NWPFrame(test)
    test.strtime_to_datetime('date', '%Y%m%d%H%M', inplace=True)
    test.datetime_to_strtime('date', '%Y-%m-%d %H:%M', inplace=True)
    df_date = test.split_strcol(
        'date', ['year', 'month', 'day', 'hour', 'min'], r'[-\s:]'
    )[['month', 'day', 'hour', 'min']].astype(int)
    test = pd.concat([df_date, test], axis=1)

    fs = skyds.get_init_features()
    target = skyds.get_init_target()

    train = train[fs + target]
    test = test[fs + target]

    train = train[(train['month'] == 1) | (train['month'] == 2)]
    test = test[(test['month'] == 1) | (test['month'] == 2)]

    X = train.iloc[:, :-1]
    y = train.iloc[:, -1]

    ss = StandardScaler()
    X = ss.fit_transform(X)
    y = y.values

    X, y = skyds.convert.balanced(X, y)

    spX, spy = skyds.convert.split_blocks(X, y, n_folds=5)
    print(spX)

    spX, spy = preprocess.split(X, y, n_folds=5)
    X = pd.concat([spX[n] for n in spX if n != 0]).reset_index(drop=True)
    y = pd.concat([spy[n] for n in spy if n != 0]).reset_index(drop=True)

    X_test = spX[0].reset_index(drop=True)
    y_test = spy[0].reset_index(drop=True)

    from sklearn.ensemble import RandomForestClassifier
    clf1 = RandomForestClassifier(max_features=2)
    clf2 = SkySVM()
    meta = LogisticRegression()

    # 学習
    # (注)balancedしてない
    sta = SkyStacking((clf1, clf2), meta)
    sta.fit(X, y)
    p = sta.predict(X_test)

    clf1.fit(X.values, y.values[:, 0])
    print(np.array(X.keys())[np.argsort(clf1.feature_importances_)[::-1]])
    p_rf = clf1.predict(X_test.values)

    # mlxtendのstacking
    sc = StackingClassifier(classifiers=[clf1, clf2], meta_classifier=meta)
    sc.fit(X.values, y.values[:, 0])
    p_sc = sc.predict(X_test.values)

    y_test = np.where(y_test.values[:, 0] > 1, 0, 1)
    p = np.where(p > 1, 0, 1)
    p_rf = np.where(p_rf > 1, 0, 1)
    p_sc = np.where(p_sc > 1, 0, 1)

    f1 = f1_score(y_true=y_test, y_pred=p)
    print("stacking", f1)

    f1_rf = f1_score(y_true=y_test, y_pred=p_rf)
    print("random forest", f1_rf)

    f1_sc = f1_score(y_true=y_test, y_pred=p_sc)
    print("stacked classifier", f1_sc)


if __name__ == "__main__":
    main()
