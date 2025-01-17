def main():
    import pickle
    import matplotlib.pyplot as plt
    import skynet.datasets as skyds
    from skynet import DATA_DIR
    from sklearn.preprocessing import StandardScaler

    icao = 'RJFK'
    save_dir = '%s/ARC-common/fit_output/JMA_MSM/vis' % DATA_DIR
    save_name = 'GLOBAL_METAR-%s.vis' % icao
    month_keys = ['month:1-2', 'month:3-4', 'month:5-6', 'month:7-8', 'month:9-10', 'month:11-12']
    clfs = {key: [] for key in month_keys}
    n_clfs = [4, 16, 80]
    mlalgos = ['stacking', 'boosting', 'forest']
    for n_clf, mlalgo in zip(n_clfs, mlalgos):
        model_dir = '/Users/makino/PycharmProjects/SkyCC/trained_models/%s/%s' % (icao, mlalgo)

        for key in month_keys:
            clfs[key] += [
                pickle.load(open("%s/%s/clf%03d.pkl" % (model_dir, key, i), "rb"))
                for i in range(n_clf)
            ]

    pickle.dump(clfs, open('%s/%s.pkl' % (save_dir, save_name), 'wb'))


if __name__ == '__main__':
    main()
