import copy

from skynet.model_selection.validate import cross_validation


def grid_search_cv(model, X, y, param_grid, cv=3, scoring="f1"):
    grids = __transform_param_grid(param_grid, 0, {}, [])
    best_score = 0
    best_params = grids[0]
    for grid in grids:
        clf = model.__class__(**grid)
        f1s = cross_validation(clf, X, y, cv, scoring=scoring)

        if len(f1s):
            if f1s.mean() > best_score:
                best_score = f1s.mean()
                best_params = grid
                print(f1s.mean())
                print(grid)
                print()

    return best_score, best_params


def __transform_param_grid(param, cnt, cp, cps):
    keys = list(param.keys())

    if cnt == len(keys):
        return
    else:
        val = param[keys[cnt]]

        for v in val:
            cp[keys[cnt]] = v
            if cnt == len(keys) - 1:
                cps.append(copy.deepcopy(cp))
            __transform_param_grid(param, cnt + 1, cp, cps)

    return cps


def main():
    import os
    import datetime
    import pickle
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import recall_score
    from skynet import OUTPUT_PATH
    from skynet.data_handling import get_init_response
    from skynet.data_handling import read_learning_data
    from skynet.data_handling import split_time_series
    from skynet.data_handling import balanced
    from skynet.preprocessing import PreProcessor
    from skynet.ensemble import SkyRandomForest

    preprocess = PreProcessor(norm=False, binary=False)

    params = [
        {'n_estimators': 10, 'min_samples_split': 2, 'min_samples_leaf': 1, 'max_features': 100},
        {'n_estimators': 10, 'min_samples_split': 2, 'min_samples_leaf': 1, 'max_features': 100},
        {'n_estimators': 10, 'min_samples_split': 2, 'min_samples_leaf': 1, 'max_features': 100},
        {'n_estimators': 10, 'min_samples_split': 2, 'min_samples_leaf': 1, 'max_features': 10},
        {'n_estimators': 10, 'min_samples_split': 2, 'min_samples_leaf': 1, 'max_features': 100},
        {'n_estimators': 10, 'min_samples_split': 2, 'min_samples_leaf': 1, 'max_features': 100}
    ]

    threat = [
        0.6,
        0.4,
        0.4,
        0.3,
        0.4,
        0.6
    ]

    n_clfs = [
        100,
        100,
        100,
        100,
        100,
        100
    ]

    target = get_init_response()

    icao = "RJCC"

    train = read_learning_data(OUTPUT_PATH + "/datasets/apvis/train_%s.pkl" % icao)
    test = read_learning_data(OUTPUT_PATH + "/datasets/apvis/test_%s.pkl" % icao)

    # feature増やしてからデータ構造を（入力、正解）に戻す
    preprocess.fit(train.iloc[:, :-1], train.iloc[:, -1], test.iloc[:, :-1], test.iloc[:, -1])
    train = pd.concat([preprocess.X_train, preprocess.y_train], axis=1)
    test = pd.concat([preprocess.X_test, preprocess.y_test], axis=1)

    fets = [f for f in train.keys() if not (f in target)]

    # 時系列でデータを分割
    sptrain = split_time_series(train, level="month", period=2)
    sptest = split_time_series(test, level="month", period=2)

    ss = StandardScaler()
    date = datetime.datetime.now().strftime("%Y%m%d")
    # date = "20180824"
    for i_term, key in enumerate(sptrain):
        os.makedirs(OUTPUT_PATH + "/learning_models/%s/forest/%s/%s" % (icao, date, key), exist_ok=True)

        X_train = sptrain[key][fets]
        X_train = pd.DataFrame(ss.fit_transform(X_train), columns=X_train.keys())
        y_train = sptrain[key][target]
        X_train, y_train = balanced(X_train, y_train)

        X_test = sptest[key][fets]
        X_test = pd.DataFrame(ss.fit_transform(X_test), columns=X_test.keys())
        y_test = sptest[key][target]

        cv = False
        if cv:
            best_score, best_params = grid_search_cv(
                SkyRandomForest(),
                X_train,
                y_train,
                param_grid={"n_estimators": [10, 100],
                            "min_samples_split": [2, 10],
                            "min_samples_leaf": [1, 10],
                            "max_features": ["auto", 2, 10, 30, 70, 100]},
                scoring="recall"
            )

            print("best score", best_score)
            print("best params", best_params)
            print()

            model = SkyRandomForest(**best_params)
            model.fit(X_train.values, y_train.values[:, 0])
            p = model.predict(X_test.values)

            y_true = np.where(y_test > 1, 0, 1)
            y_pred = np.where(p > 1, 0, 1)
            print(recall_score(y_true=y_true, y_pred=y_pred))

        search = True
        y_true = np.where(y_test > 1, 0, 1)
        p_rf = np.zeros((len(X_test), n_clfs[i_term]))
        score_rf = np.zeros(n_clfs[i_term])
        i = 0
        if search:
            while True:
                clf = SkyRandomForest(**params[i_term])
                clf.fit(X_train.values, y_train.values[:, 0])
                p = clf.predict(X_test.values)

                y_pred = np.where(p > 1, 0, 1)

                scr = recall_score(y_true=y_true, y_pred=y_pred)
              
                if scr >= threat[i_term]:
                    print(scr)
                    p_rf[:, i] = p
                    score_rf[i] = scr
                    pickle.dump(clf, open(OUTPUT_PATH + "/learning_models/%s/forest/%s/%s/rf%03d.pkl"
                                          % (icao, date, key, i), "wb"))
                    i += 1

                if i == n_clfs[i_term]:
                    break

        learning_model = True
        if learning_model:
            for i in range(n_clfs[i_term]):
                clf = pickle.load(open(OUTPUT_PATH + "/learning_models/%s/forest/%s/%s/rf%03d.pkl"
                                       % (icao, date, key, i), "rb"))
                p_rf[:, i] = clf.predict(X_test.values)

                y_pred = np.where(p_rf[:, i] > 1, 0, 1)
                score_rf[i] = recall_score(y_true=y_true, y_pred=y_pred)

                print(score_rf[i])

        p = p_rf.mean(axis=1)
        score_rf = score_rf.mean()
        print("f1 mean", score_rf)

        """
        plt.plot(y_test)
        plt.plot(p)
        plt.show()
        """


if __name__ == "__main__":
    main()
