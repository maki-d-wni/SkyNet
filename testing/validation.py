def test1():
    import pickle
    import matplotlib.pyplot as plt
    import skynet.datasets as skyds
    import skynet.testing as skytest
    from skynet import DATA_DIR
    from skynet.mlcore.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler

    icao = 'RJOT'
    model_dir = '%s/ARC-common/fit_output/JMA_MSM/vis' % DATA_DIR
    model_name = 'GLOBAL_METAR-%s.vis.dev' % icao
    data_dir = '%s/ARC-common/fit_input/JMA_MSM/vis' % DATA_DIR
    data_name = 'GLOBAL_METAR-%s.vis' % icao

    clfs = pickle.load(open('%s/%s.pkl' % (model_dir, model_name), 'rb'))

    data = skyds.read_csv('%s/%s.csv' % (data_dir, data_name))

    spX, spy = skytest.preprocessing.test1(data)

    print(spX)

    for key, clf in clfs.items():
        X = spdata[key].iloc[:, :-1]
        y = spdata[key].iloc[:, -1]

        ss = StandardScaler()

        X = ss.fit_transform(X)
        y = y.values

        p = clf.predict(X)

        plt.figure()
        plt.plot(y)
        plt.plot(p)
    plt.show()


def test2():
    pass


def main():
    test1()


if __name__ == '__main__':
    main()
