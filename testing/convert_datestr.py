def main():
    import pickle
    import matplotlib.pyplot as plt
    import pandas as pd
    import skynet.datasets as skyds
    from skynet import DATA_DIR
    from skynet.nwp2d import NWPFrame

    icao = 'RJCC'
    data_dir = '%s/ARC-common/fit_input/JMA_MSM/vis' % DATA_DIR
    data_name = 'GLOBAL_METAR-%s.vis' % icao

    data = skyds.read_csv('%s/%s.csv' % (data_dir, data_name))
    # data.drop(['year', 'month', 'day', 'hour', 'min'], axis=1, inplace=True)

    '''
    data = NWPFrame(data)
    data.strtime_to_datetime(date_key='date', fmt='%Y-%m-%d %H:%M', inplace=True)
    data.datetime_to_strtime(date_key='date', fmt='%Y%m%d%H%M', inplace=True)
    '''

    data = NWPFrame(data)
    data.strtime_to_datetime(date_key='date', fmt='%Y%m%d%H%M', inplace=True)
    data.datetime_to_strtime(date_key='date', fmt='%Y-%m-%d %H:%M', inplace=True)
    df_date = data.split_strcol('date', ['year', 'month', 'day', 'hour', 'min'], pattern='[-: ]')
    df_date = df_date[['year', 'month', 'day', 'hour', 'min']]

    data = pd.concat([df_date, data], axis=1)
    print(data)

    # data.to_csv('%s/%s.csv' % (data_dir, data_name), index=False)


if __name__ == '__main__':
    main()
