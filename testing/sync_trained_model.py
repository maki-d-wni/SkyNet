def main():
    import pickle
    import matplotlib.pyplot as plt
    import skynet.datasets as skyds
    from skynet import DATA_DIR
    from sklearn.preprocessing import StandardScaler

    icao = 'RJOT'
    model_dir = '/Users/makino/PycharmProjects/SkyCC/trained_models/%s/forest' % icao
    month_keys = ['month:1-2', 'month:3-4', 'month:5-6', 'month:7-8', 'month:9-10', 'month:11-12']
    save_dir = '%s/ARC-common/fit_output/JMA_MSM/vis' % DATA_DIR
    save_name = 'GLOBAL_METAR-%s.vis' % icao

    clfs = {}
    for key in month_keys:
        clfs[key] = [
            pickle.load(open("%s/%s/rf%03d.pkl" % (model_dir, key, i), "rb"))
            for i in range(100)
        ]

    pickle.dump(clfs, open('%s/%s.pkl' % (save_dir, save_name), 'wb'))


if __name__ == '__main__':
    main()
