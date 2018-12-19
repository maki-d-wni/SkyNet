import os
import re
import requests
import pymongo
import pickle
import datetime
from collections import OrderedDict
from bs4 import BeautifulSoup

WEB_PATH = '/home/maki-d/PycharmProjects/SimRec/web'

NWP_INFO = {
    '400220000': {
        'nwp': 'MSM',
        'layer': 'surface',
        'base time': '00',
        'validity time': '0015'
    },
    '400220001': {
        'nwp': 'MSM',
        'layer': 'surface',
        'base time': '03',
        'validity time': '0015'
    },
    '400220002': {
        'nwp': 'MSM',
        'layer': 'surface',
        'base time': '06',
        'validity time': '0015'
    },
    '400220003': {
        'nwp': 'MSM',
        'layer': 'surface',
        'base time': '09',
        'validity time': '0015'
    },
    '400220004': {
        'nwp': 'MSM',
        'layer': 'surface',
        'base time': '12',
        'validity time': '0015'
    },
    '400220005': {
        'nwp': 'MSM',
        'layer': 'surface',
        'base time': '15',
        'validity time': '0015'
    },
    '400220006': {
        'nwp': 'MSM',
        'layer': 'surface',
        'base time': '18',
        'validity time': '0015'
    },
    '400220007': {
        'nwp': 'MSM',
        'layer': 'surface',
        'base time': '21',
        'validity time': '0015'
    },
    '400220008': {
        'nwp': 'MSM',
        'layer': 'upper',
        'base time': '00',
        'validity time': '0015'
    },
    '400220009': {
        'nwp': 'MSM',
        'layer': 'upper',
        'base time': '03',
        'validity time': '0015'
    },
    '400220010': {
        'nwp': 'MSM',
        'layer': 'upper',
        'base time': '06',
        'validity time': '0015'
    },
    '400220011': {
        'nwp': 'MSM',
        'layer': 'upper',
        'base time': '09',
        'validity time': '0015'
    },
    '400220012': {
        'nwp': 'MSM',
        'layer': 'upper',
        'base time': '12',
        'validity time': '0015'
    },
    '400220013': {
        'nwp': 'MSM',
        'layer': 'upper',
        'base time': '15',
        'validity time': '0015'
    },
    '400220014': {
        'nwp': 'MSM',
        'layer': 'upper',
        'base time': '18',
        'validity time': '0015'
    },
    '400220015': {
        'nwp': 'MSM',
        'layer': 'upper',
        'base time': '21',
        'validity time': '0015'
    },
    '400220115': {
        'nwp': 'MSM',
        'layer': 'surface',
        'base time': '03',
        'validity time': '1633'
    },
    '400220123': {
        'nwp': 'MSM',
        'layer': 'upper',
        'base time': '03',
        'validity time': '1633'
    },
    '400220143': {
        'nwp': 'GSM',
        'layer': 'surface',
        'base time': '12',
        'validity time': '0084'
    },
    '400220148': {
        'nwp': 'GSM',
        'layer': 'upper',
        'base time': '12',
        'validity time': '0084'
    }
}

GSM_INFO = {

}


class NwpDB(object):
    client = pymongo.MongoClient()

    def __init__(self, name=None):
        self.db = self.client[name]

    def register_one_document(self, document, collection):
        self.db[collection].insert_one(document)

    def register_documents(self, documents, collection):
        self.db[collection].insert_many(documents)


class MSMDBBase(NwpDB):
    name = 'MSM'
    collection_names = []
    params = {
        'surface': [
            'Pressure reduced to MSL',
            'u-component of wind',
            'v-component of wind',
            'Temperature',
            'Relative humidity',
            'Low cloud cover',
            'Medium cloud cover',
            'High cloud cover',
            'Total cloud cover',
            'Total precipitation'
        ],
        'upper': [
            'Geopotential height',
            'Relative humidity',
            'Temperature',
            'Vertical velocity [pressure]',
            'u-component of wind',
            'v-component of wind'
        ]
    }
    level = {
        'surface': ['surface'],
        'upper': [
            '300',
            '400',
            '500',
            '600',
            '700',
            '800',
            '850',
            '900',
            '925',
            '950',
            '975',
            '1000'
        ]
    }
    base_time = {
        'surface': ['%02d' % t for t in range(0, 24, 3)],
        'upper': ['%02d' % t for t in range(3, 24, 6)]
    }
    validity_time = {
        'surface': ['%02d' % t for t in range(40)],
        'upper': ['%02d' % t for t in range(0, 40, 3)]
    }
    shape = {
        'surface': (505, 481),
        'upper': (253, 241)
    }

    for lay in ['surface', 'upper']:
        for p in params[lay]:
            for l in level[lay]:
                for bt in base_time[lay]:
                    for vt in validity_time[lay]:
                        collection_names += [
                            '%s_%s_bt%s_vt%s' % (p, l, bt, vt)
                        ]

    def __init__(self):
        super(MSMDBBase, self).__init__(name=self.name)


class MSMDB(MSMDBBase):
    def read_documents(self, query, collection):
        docs = self.db[collection].find(query)

        return docs


def make_msm_docs(layer, param, level, bt, vt, body_path):
    name = '%s_%s_bt%s_vt%s' % (param, level, bt, vt)

    if layer == 'surface':
        path = '%s/%s.pkl' % (body_path, param)
    else:
        path = '%s/%s_%s.pkl' % (body_path, param, level)

    values = pickle.load(open(path, 'rb'))
    docs = [
        OrderedDict({
            'date': i,
            'values': pickle.dumps(v)
        })
        for i, v in zip(values.index, values.values)
    ]
    print(name)
    return docs


def get_today_link(url):
    if re.match('\d{8}_\d{6}.\d{3}', url.split(sep='/')[-1]):
        ele = url.split('/')
        file_date = ele[-1][:8]

        today = datetime.datetime.now()
        today = today - datetime.timedelta(hours=12)
        today = str(today.date())
        today = ''.join(today.split('-'))
        if file_date == today:
            return url
        else:
            f = open('%s/tmp/err/err.log' % WEB_PATH, 'a')
            f.write('%s url = %s : today = %s \n'
                    % (str(datetime.datetime.now()), url, today))
            f.close()
    else:
        url_info = requests.get(url)
        html = url_info.content
        soup = BeautifulSoup(html, 'html.parser')
        link = None
        for a in soup.find_all('a'):
            link = a.get('href')

        url = 'http://stock1.wni.co.jp' + link
        url = get_today_link(url)

        return url


def main():
    import argparse
    import gc
    import pygrib

    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--get', action='store_true')
    parser.add_argument('-a', '--add', action='store_true')
    parser.add_argument('--tagid')
    parser.add_argument('--date')
    parser.add_argument('-c', '--collections', action='store_true')
    args = parser.parse_args()

    msm = MSMDB()

    tag_id = args.tagid

    if args.date == 'now':
        url = 'http://stock1.wni.co.jp/cgi-bin/list.cgi?path=/stock_hdd/%s' % tag_id
        url = get_today_link(url)
    else:
        url = ''

    nwp_type = NWP_INFO[tag_id]['nwp']
    layer = NWP_INFO[tag_id]['layer']
    bt = NWP_INFO[tag_id]['base time']
    vt = NWP_INFO[tag_id]['validity time']
    date_dir = url.split(sep='/')[-1]

    save_path = '/home/ai-corner/part1/%s/%s/bt%s/vt%s/%s' % (
        nwp_type, layer, bt, vt, date_dir
    )
    os.makedirs(save_path, exist_ok=True)

    if args.get:
        if os.path.exists(save_path + '/' + url.split(sep='/')[-1]):
            print('file already exist')
        else:
            os.system('wget %s -P %s ' % (url, save_path))

    elif args.collections:
        print(list(msm.db.list_collection_names()))

    elif args.add:
        if args.date == 'now':
            date = ''.join(str(datetime.date.today()).split(sep='-'))
            grbs = pygrib.open(save_path + '/' + url.split(sep='/')[-1])
            collections = msm.db.list_collection_names()
            # level check
            collections = [c for c in collections if c.split('_')[1] in msm.level[layer]]
            # base time check
            collections = [c for c in collections if c.split('_')[2][2:] == bt]
            # validity time check
            collections = [c for c in collections if c.split('_')[3][2:] == '00']

            pattern = re.compile('\d+')
            for collection in collections:
                grb_args = collection.split('_')
                param = grb_args[0]
                level = grb_args[1]
                forecast_time = int(pattern.findall(grb_args[3])[0])

                if level == layer:
                    grb = grbs.select(forecastTime=forecast_time, parameterName=param)[0]
                else:
                    grb = grbs.select(forecastTime=forecast_time, parameterName=param, level=int(level))[0]
                v = grb.values.reshape(-1)

                doc = OrderedDict(
                    {
                        'date': date,
                        'values': pickle.dumps(v)
                    }
                )

                if list(msm.db[collection].find({'date': date})):
                    print('already exist')
                else:
                    print(date, collection, level, forecast_time)
                    msm.register_one_document(doc, collection)

                del grb, doc
                gc.collect()

    run = False
    if run:
        # kokokara
        msm = MSMDB()
        for layer in ['surface', 'upper']:
            msm.base_time[layer] = ['03']
            msm.validity_time[layer] = ['00']

        body_path = '/home/maki-d/PycharmProjects/SimRec/database/FT0'
        for key in ['surface', 'upper']:
            for p in msm.params[key]:
                for l in msm.level[key]:
                    for bt in msm.base_time[key]:
                        for vt in msm.validity_time[key]:
                            name = '%s_%s_bt%s_vt%s' % (p, l, bt, vt)

                            if key == 'surface':
                                path = '%s/%s.pkl' % (body_path, p)
                            else:
                                path = '%s/%s_%s.pkl' % (body_path, p, l)

                            values = pickle.load(open(path, 'rb'))
                            docs = [
                                OrderedDict({
                                    'date': i,
                                    'values': pickle.dumps(v)
                                })
                                for i, v in zip(values.index, values.values)
                            ]
                            msm.register_documents(docs, collection=name)
                            print(name)
        # kokomade register msm to database


if __name__ == '__main__':
    main()
