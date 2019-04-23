def main():
    import pickle
    import matplotlib.pyplot as plt
    import skynet.datasets as skyds
    from skynet import DATA_DIR
    from sklearn.preprocessing import StandardScaler

    icao = 'RJOT'
    model_dir = '%s/ARC-common/fit_output/JMA_MSM/vis' % DATA_DIR
    model_name = 'GLOBAL_METAR-%s.vis.dev' % icao
    data_dir = '%s/skynet' % DATA_DIR
    data_name = 'test_%s' % icao

    clfs = pickle.load(open('%s/%s.pkl' % (model_dir, model_name), 'rb'))

    test = skyds.read_csv('%s/%s.csv' % (data_dir, data_name))
    fets = skyds.get_init_features()
    target = skyds.get_init_target()

    test = test[fets + target]
    sptest = skyds.convert.split_time_series(test, test['month'], date_fmt='%m')

    for key, clf in clfs.items():
        X = sptest[key].iloc[:, :-1]
        y = sptest[key].iloc[:, -1]

        ss = StandardScaler()

        X = ss.fit_transform(X)
        y = y.values

        p = clf.predict(X)

        plt.figure()
        plt.plot(y)
        plt.plot(p)
    plt.show()


if __name__ == '__main__':
    main()
