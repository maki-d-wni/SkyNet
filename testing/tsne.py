def test1():
    import matplotlib.pyplot as plt
    import skynet.datasets as skyds
    from skynet import DATA_DIR
    from sklearn.preprocessing import StandardScaler
    from sklearn.manifold import TSNE

    icao = 'RJOT'
    data_dir = '%s/ARC-common/fit_input/JMA_MSM/vis' % DATA_DIR
    data_name = 'GLOBAL_METAR-%s.vis' % icao

    data = skyds.read_csv('%s/%s.csv' % (data_dir, data_name))
    fets = skyds.get_init_features()
    target = skyds.get_init_target()

    data = data[fets + target]
    spdata = skyds.convert.split_time_series(data, data['month'], date_fmt='%m')
