import os
import pickle
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from mlxtend.classifier import StackingClassifier
from sklearn.metrics import recall_score
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier


def fit_n_models(clf, n_clfs, X_train, y_train, X_test, y_test, save_dir):
    search = True
    y_true = np.where(y_test > 1, 0, 1)
    p_n = np.zeros((len(X_test), n_clfs))
    score = np.zeros(n_clfs)
    i = 0
    if search:
        while True:
            if clf == 'stacking':
                # 第一層のクラシファイア
                clf_children = [
                                   GaussianNB() for _ in range(4)
                               ] + [
                                   RandomForestClassifier() for _ in range(4)
                               ] + [
                                   GradientBoostingClassifier() for _ in range(4)
                               ]

                # メタクラシファイア
                clf_meta = LogisticRegression()
                clf = StackingClassifier(
                    classifiers=clf_children,
                    meta_classifier=clf_meta,
                    use_probas=True
                )
            elif clf == 'naive_bayes':
                clf = GaussianNB()
            elif clf == 'forest':
                clf = RandomForestClassifier()
            elif clf == 'boosting':
                clf = GradientBoostingClassifier()

            clf.fit(X_train.values, y_train.values[:, 0])
            p = clf.predict(X_test.values)

            y_pred = np.where(p > 1, 0, 1)
            p_n[:, i] = p
            score[i] = recall_score(y_true=y_true, y_pred=y_pred)

            print('recall score :', score[i])

            pickle.dump(clf, open("%s/clf%03d.pkl"
                                  % (save_dir, i), "wb"))
            i += 1

            if i == n_clfs:
                break
    return p_n, score


def predict_n_models(n_clfs, X, y, model_dir):
    y_true = np.where(y > 1, 0, 1)
    p_n = np.zeros((len(X), n_clfs))
    score = np.zeros(n_clfs)
    for i in range(n_clfs):
        clf = pickle.load(open("%s/clf%03d.pkl"
                               % (model_dir, i), "rb"))
        p_n[:, i] = clf.predict(X.values)

        y_pred = np.where(p_n[:, i] > 1, 0, 1)
        score[i] = recall_score(y_true=y_true, y_pred=y_pred)

        print(score[i])
    return score


def main():
    import matplotlib.pyplot as plt
    import skynet.datasets as skyds
    from sklearn.preprocessing import StandardScaler
    from skynet import USER_DIR, DATA_DIR
    from skynet.datasets import convert

    n_clfs = [
        10,
        10,
        10,
        10,
        10,
        10
    ]

    target = skyds.get_init_target()

    icao = 'RJFK'
    # 'RJSS',
    # 'RJTT',
    # 'ROAH',
    # 'RJOC',
    # 'RJOO',
    # 'RJCH',
    # 'RJFF',
    # 'RJFK',
    # 'RJGG',
    # 'RJNK',
    # 'RJOA',
    # 'RJOT',

    mlalgo = 'stacking'

    data_dir = '%s/ARC-common/fit_input/JMA_MSM/vis' % DATA_DIR
    data_name = 'GLOBAL_METAR-%s.vis' % icao
    train = skyds.read_csv('%s/%s.csv' % (data_dir, data_name))
    test = skyds.read_csv('%s/skynet/test_%s.csv' % (DATA_DIR, icao))

    # 時系列でデータを分割
    sptrain = convert.split_time_series(train, train['date'], level="month", period=2)
    sptest = convert.split_time_series(test, test['date'], level="month", period=2)

    ss = StandardScaler()
    model_dir = '%s/PycharmProjects/SkyCC/trained_models' % USER_DIR

    period_keys = [
        'month:1-2',
        'month:3-4',
        'month:5-6',
        'month:7-8',
        'month:9-10',
        'month:11-12'
    ]

    init_fets = skyds.get_init_features(code='long')
    for i_term, key in enumerate(period_keys):
        os.makedirs(
            '%s/%s/%s/%s'
            % (model_dir, icao, mlalgo, key), exist_ok=True
        )

        # fets = pearson_correlation(sptrain[key][init_fets], sptrain[key][target], depth=30)
        fets = init_fets

        X_train = sptrain[key][fets]
        X_train = pd.DataFrame(ss.fit_transform(X_train), columns=X_train.keys())
        y_train = sptrain[key][target]
        X_train, y_train = convert.balanced(X_train, y_train)

        X_test = sptest[key][fets]
        X_test = pd.DataFrame(ss.fit_transform(X_test), columns=X_test.keys())
        y_test = sptest[key][target]

        save_dir = "%s/%s/%s/%s" % (model_dir, icao, mlalgo, key)
        p_n, score = fit_n_models(mlalgo, n_clfs[i_term], X_train, y_train, X_test, y_test, save_dir)

        p = p_n.mean(axis=1)
        score = score.mean()
        print("f1 mean", score)

        plt.figure()
        plt.plot(y_test)
        plt.plot(p)
    plt.show()


if __name__ == "__main__":
    main()
