def test1(X_train, y_train, X_test, y_test):
    import matplotlib.pyplot as plt
    from sklearn.model_selection import cross_val_score
    from sklearn.metrics import accuracy_score
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.neural_network import MLPClassifier
    from mlxtend.classifier import StackingClassifier

    # 第一層のクラシファイア
    clf_children = [
        KNeighborsClassifier(n_neighbors=1),
        RandomForestClassifier(),
        GaussianNB()
    ]

    clf_children2 = [
        RandomForestClassifier() for _ in range(100)
    ]

    clf_children3 = [
        GaussianNB() for _ in range(16)
    ]

    clf_children4 = [
                        KNeighborsClassifier(n_neighbors=1) for _ in range(2)
                    ] + [
                        RandomForestClassifier() for _ in range(2)
                    ] + [
                        GaussianNB() for _ in range(16)
                    ]

    # メタクラシファイア
    clf_meta = LogisticRegression()
    clf_meta2 = SVC(probability=True)
    clf_meta3 = GaussianNB()
    clf_meta4 = MLPClassifier()

    clf_stack = StackingClassifier(
        classifiers=clf_children3,
        meta_classifier=clf_meta,
        use_probas=True
    )

    clf_names = [
        'KNN',
        'Random Forest',
        'Naive Bayes',
        'Stacking'
    ]

    # 実験
    for clf, clf_name in zip(clf_children[-1:] + [clf_stack], clf_names[-2:]):
        scores = cross_val_score(clf, X_train, y_train, cv=3, scoring='accuracy')
        print("Validation Accuracy: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), clf_name))

        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        # print('probability')
        c = clf.predict_proba(X_test)

        acc = accuracy_score(y_test, y_pred)
        print('Test Accuracy: %0.2f [%s]' % (acc, clf_name))
        print()

        plt.figure()
        plt.title(clf_name)
        plt.plot(y_test)
        plt.plot(y_pred)

    return clf_stack


def main():
    import pickle
    import matplotlib.pyplot as plt
    import pandas as pd
    import skynet.datasets as skyds
    import skynet.testing as skytest
    from skynet import DATA_DIR
    from skynet.mlcore.feature_selection.filter import pearson_correlation

    # icao = 'RJFK'
    # icao = 'RJFT'
    # icao = 'RJOT'
    # icao = 'RJCC'
    icao = 'RJAA'

    data_dir = '%s/ARC-common/fit_input/JMA_MSM/vis' % DATA_DIR
    model_dir = '%s/ARC-common/fit_output/JMA_MSM/vis' % DATA_DIR
    model_name = 'GLOBAL_METAR-%s.vis' % icao
    data_name = 'GLOBAL_METAR-%s.vis' % icao
    month_keys = ['month:1-2', 'month:3-4', 'month:5-6', 'month:7-8', 'month:9-10', 'month:11-12']

    # トレーニングデータの準備
    train = skyds.read_csv('%s/%s.csv' % (data_dir, data_name))
    spX_train, spy_train = skytest.preprocessing.normal(train)

    # -- テストデータの準備
    test = pd.read_csv('/Users/makino/PycharmProjects/SkyCC/data/skynet/test_%s.csv' % icao, sep=',')
    spX_test, spy_test = skytest.preprocessing.normal(test)

    clfs = {}
    for key in month_keys:
        fets = pearson_correlation(spX_train[key], spy_train[key], depth=30)

        X_train, y_train = spX_train[key][fets].values, spy_train[key].values
        X_train, y_train = skyds.convert.balanced(X_train, y_train)
        X_test, y_test = spX_test[key][fets].values, spy_test[key].values

        clfs[key] = test1(X_train, y_train, X_test, y_test)

    pickle.dump(clfs, open('%s/%s.pkl' % (model_dir, model_name), 'wb'))

    plt.show()


if __name__ == '__main__':
    main()
