def main():
    import pandas as pd
    from skynet import DATA_DIR

    header_path = '%s/text/metar/head.txt' % DATA_DIR
    f = open(header_path)
    header = f.read().split(sep=',')
    f.close()

    icao = 'RJSK'
    '''
    'RJOT',
    'RJAA',
    'RJSC',
    'RJSI',
    'RJSK',
    'RJSM',
    'RJSN',
    'RJSS',
    'RJTT',
    'ROAH',
    'RJOC',
    'RJOO',
    # 'RJBB',
    'RJCC',
    'RJCH',
    'RJFF',
    'RJFK',
    'RJGG',
    'RJNK',
    'RJOA',
    '''

    data_path = '%s/text/metar/2017/%s.txt' % (DATA_DIR, icao)

    metar = pd.read_csv(data_path, names=header)
    metar.drop_duplicates('date', keep='first', inplace=True)
    metar['date'] = [str(d)[:12] for d in metar['date']]
    metar.replace(-1, 9999, inplace=True)

    import numpy as np
    print(np.unique(metar['visibility']))

    import matplotlib.pyplot as plt
    plt.hist(metar['visibility'])
    plt.show()

    metar = metar[['date', 'visibility']].reset_index(drop=True)

    metar.to_csv('%s/csv/metar/airport_vis/metar_%s.csv' % (DATA_DIR, icao), index=False)


if __name__ == '__main__':
    main()
