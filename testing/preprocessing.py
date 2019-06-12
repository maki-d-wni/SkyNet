def normal(data):
    import pandas as pd
    import skynet.datasets as skyds
    from sklearn.preprocessing import StandardScaler

    fets = skyds.get_init_features()
    target = skyds.get_init_target()

    data = data[fets + target]
    date = data['month']

    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    ss = StandardScaler()
    X = pd.DataFrame(ss.fit_transform(X.values), columns=X.keys())

    spX = skyds.convert.split_time_series(X, date, date_fmt='%m')
    spy = skyds.convert.split_time_series(y, date, date_fmt='%m')

    return spX, spy
