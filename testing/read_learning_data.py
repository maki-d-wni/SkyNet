def test1():
    import pickle
    import pandas as pd
    import skynet.nwp2d as npd
    import skynet.datasets as skyds

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

    X_test = X_test[(X_test['month'] == start_month) | (X_test['month'] == end_month)]
    y_test = y_test.loc[X_test.index]

    ss = StandardScaler()
    X_test = ss.fit_transform(X_test)
    y_test = y_test.values
