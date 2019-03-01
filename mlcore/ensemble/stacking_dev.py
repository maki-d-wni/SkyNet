def main():
    import pickle
    import matplotlib.pyplot as plt
    import pandas as pd
    import skynet.nwp2d as npd
    import skynet.datasets as skyds
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import cross_val_score
    from sklearn.metrics import accuracy_score
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.ensemble import RandomForestClassifier
    from mlxtend.classifier import StackingClassifier
    from skynet import DATA_DIR

    # トレーニングデータの準備
    icao = 'RJOT'
    data_dir = '%s/MSM/airport.process' % DATA_DIR

    data = pd.read_csv('%s/%s.csv' % (data_dir, icao), sep=',')
    data = data.iloc[:, 1:]

    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    X = X[(X['month'] == 1) | (X['month'] == 2)]
    y = y.loc[X.index]

    ss = StandardScaler()
    X = ss.fit_transform(X)
    y = y.values

    plt.figure()
    plt.hist(y)

    X, y = skyds.convert.balanced(X, y)

    plt.figure()
    plt.hist(y)
    # plt.show()

    # 第一層のクラシファイア
    clf_children = [
        KNeighborsClassifier(n_neighbors=1),
        RandomForestClassifier(),
        GaussianNB()
    ]

    clf_children2 = [
        RandomForestClassifier() for _ in range(10)
    ]

    clf_children3 = [
        GaussianNB() for _ in range(10)
    ]

    clf_children4 = [
                        KNeighborsClassifier(n_neighbors=1) for _ in range(1)
                    ] + [
                        RandomForestClassifier() for _ in range(1)
                    ] + [
                        GaussianNB() for _ in range(8)
                    ]

    # メタクラシファイア
    clf_meta = LogisticRegression()
    clf_meta2 = SVC(probability=True)
    clf_meta3 = GaussianNB()

    clf_stack = StackingClassifier(
        classifiers=clf_children3,
        meta_classifier=clf_meta,
        use_probas=True
    )

    model_names = [
        'KNN',
        'Random Forest',
        'Naive Bayes',
        'Stacking'
    ]

    # -- テストデータの準備
    test = pickle.load(open('/Users/makino/PycharmProjects/SkyCC/data/skynet/test_%s.pkl' % icao, 'rb'))
    test['date'] = test['date'].astype(int).astype(str)
    test = npd.NWPFrame(test)
    test.strtime_to_datetime('date', '%Y%m%d%H%M', inplace=True)
    test.datetime_to_strtime('date', '%Y-%m-%d %H:%M', inplace=True)
    df_date = test.split_strcol(
        'date', ['year', 'month', 'day', 'hour', 'min'], r'[-\s:]'
    )[['month', 'day', 'hour', 'min']].astype(int)
    test = pd.concat([df_date, test], axis=1)
    keys = skyds.get_init_features() + skyds.get_init_target()
    test = test[keys]

    X_test = test.iloc[:, :-1]
    y_test = test.iloc[:, -1]

    X_test = X_test[(X_test['month'] == 1) | (X_test['month'] == 2)]
    y_test = y_test.loc[X_test.index]

    ss = StandardScaler()
    X_test = ss.fit_transform(X_test)
    y_test = y_test.values

    # 実験
    for clf, model_name in zip(clf_children + [clf_stack], model_names):
        scores = cross_val_score(clf, X, y, cv=3, scoring='accuracy')
        print("Validation Accuracy: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), model_name))

        clf.fit(X, y)
        y_pred = clf.predict(X_test)

        # print('probability')
        # print(clf.predict_proba(X_test))

        acc = accuracy_score(y_test, y_pred)
        print('Test Accuracy: %0.2f [%s]' % (acc, model_name))
        print()

        plt.figure()
        plt.title(model_name)
        plt.plot(y_test)
        plt.plot(y_pred)
    plt.show()


if __name__ == '__main__':
    main()
