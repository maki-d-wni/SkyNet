import os
import re
import requests
import datetime
import pytz
import pandas as pd
from bs4 import BeautifulSoup


def get_predict_data(url, save_dir='./'):
    os.makedirs(save_dir, exist_ok=True)
    os.system('wget %s -P %s' % (url, save_dir))


def get_metar_area(url):
    os.environ['http_proxy'] = 'http://maki-d:onigiri0802@172.16.250.1:8080'
    url_info = requests.get(url)
    html = url_info.content

    soup = BeautifulSoup(html, 'html.parser')

    dict_metar = {}
    tbl = soup.find_all('td')
    for t in tbl:
        df_metar = _html_to_dataframe(r'<td>RJ.+<br/>', str(t))
        if df_metar is not None:
            dict_metar[df_metar.loc[0, 'ICAO']] = df_metar
    return dict_metar


def _html_to_dataframe(pattern, string):
    p = re.compile(pattern)
    r = p.match(string)
    df_metar = pd.DataFrame()
    idx = 0
    if r is not None:
        metar = r.group()
        if re.findall(r'<fo', metar):
            while True:
                ss = re.findall(r'<', metar)
                es = re.findall(r'>', metar)
                if len(es) == 0:
                    break
                i = metar.find(ss[0])
                e = metar.find(es[0]) + 1
                metar = metar.replace(metar[i:e], '')
        metar = metar.split()
        df_metar.loc[0, 'ICAO'] = metar[idx]
        idx += 1

        if re.match(r'<.*>', metar[idx][:3]):
            metar[idx] = metar[idx][3:]
        df_metar['date'] = metar[idx]
        idx += 1

        if len(metar) == 2:
            return df_metar

        if metar[idx] == 'AUTO':
            df_metar['observation'] = metar[idx]
            idx += 1
        else:
            df_metar['observation'] = ''

        if re.match(r'<.*', metar[idx]):
            metar[idx] = metar[idx][4:]

        if re.match(r'.*KT', metar[idx]):
            df_metar['wind speed and direction'] = metar[idx]
            idx += 1
        else:
            df_metar['wind speed and direction'] = ''

        if re.match(r'\d{3}\D\d{3}', metar[3]):
            df_metar['wind direction variation'] = metar[idx]
            idx += 1
        else:
            df_metar['wind direction variation'] = ''

        try:
            int(metar[idx])
            df_metar['visibility'] = metar[idx]
            idx += 1
        except ValueError:
            pass

        if re.match(r'R\d{2}/', metar[idx]):
            df_metar['rvr'] = metar[idx]
            idx += 1
        else:
            df_metar['rvr'] = ''

        wc = ''
        head = ['', '+', '-', 'VC']
        qs = ['', 'MI', 'BC', 'PR', 'DR', 'BL', 'SH', 'TS', 'FZ']
        pr = ['', 'DZ', 'RA', 'SN', 'SG', 'PL', 'GR', 'GS']
        vi = ['', 'BR', 'FG', 'FU', 'VA', 'DU', 'SA', 'HZ']
        ot = ['', 'PO', 'SQ', 'FC', 'SS', 'DS']
        wc_list = [h + q + p + v + o for h in head for q in qs for p in pr for v in vi for o in ot]
        while True:
            if not metar[idx] in wc_list:
                break
            wc += metar[idx] + ' '
            idx += 1
        wc = wc[:-1]
        df_metar['weather condition'] = wc

        cig = ''
        while True:
            if re.match(r'[^/].*/[^/].*', metar[idx]):
                break
            cig += metar[idx] + ' '
            idx += 1
        cig = cig[:-1]
        df_metar['ceiling'] = cig

        df_metar['temperature/dewpoint'] = metar[idx]
        idx += 1

        df_metar['pressure'] = metar[idx]
        return df_metar
    else:
        return None


def _d2h2m2z_to_y4m2d2h2d2(date, tz='UTC'):
    today = datetime.datetime.now(tz=pytz.timezone(tz))
    year = str(today.year)
    month = str(today.month)
    date = [year + month + d[0:6] for d in date]
    date = [datetime.datetime.strptime(d, '%Y%m%d%H%M') for d in date]
    date = [d.strftime('%Y-%m-%d %H:%M') for d in date]
    return date


def get_metar_point(url):
    # os.environ['http_proxy'] = 'http://maki-d:onigiri0802@172.16.250.1:8080'
    url_info = requests.get(url)
    html = url_info.content

    soup = BeautifulSoup(html, 'html.parser')
    str_html = str(soup)
    i = str_html.find('============== METAR ==============')
    e = str_html.find('<br/></td></tr></table>')

    str_html = str_html[i:e]
    tbl = str_html.split('br/>')
    df = pd.DataFrame()
    for t in tbl:
        t = t.replace('\n', ' ')
        print(t)
        metar = _html_to_dataframe(pattern=r'R.*', string=t)
        if metar is not None:
            df = pd.concat([df, metar])
    df.reset_index(drop=True, inplace=True)
    return df


def main():
    from skynet import USER_DIR

    today = datetime.datetime.now(tz=pytz.timezone('UTC')) - datetime.timedelta(hours=3)

    year = today.year
    month = today.month
    day = today.day
    bt = today.hour
    bt = bt // 3 * 3

    icaos = ['RJAA',
             'RJAF',
             'RJAH',
             'RJAW',
             'RJBB',
             'RJBD',
             'RJBE',
             'RJCB',
             'RJCC',
             'RJCH',
             'RJCK',
             'RJCM',
             'RJCO',
             'RJEC',
             'RJEO',
             'RJER',
             'RJFF',
             'RJFG',
             'RJFK',
             'RJFM',
             'RJFO',
             'RJFR',
             'RJFS',
             'RJFT',
             'RJFU',
             'RJGG',
             'RJKA',
             'RJKN',
             'RJNA',
             'RJNG',
             'RJNK',
             'RJNS',
             'RJNT',
             'RJOA',
             'RJOB',
             'RJOC',
             'RJOH',
             'RJOK',
             'RJOM',
             'RJOO',
             'RJOS',
             'RJOT',
             'RJSA',
             'RJSC',
             'RJSI',
             'RJSK',
             'RJSM',
             'RJSN',
             'RJSS',
             'RJSY',
             'RJTT',
             'ROAH',
             'RODN',
             'ROIG',
             'ROMY'
             ]

    for icao in icaos:
        data_name = 'GLOBAL_METAR-%s.vis' % icao

        url_pred = 'http://pt-compass-arc.wni.co.jp/data/ARC-pred/pred_output/JMA_MSM/' \
                   '%04d/%02d/%02d/%02d/vis/%s.csv' % (year, month, day, bt, data_name)

        save_dir = '/%s/PycharmProjects/SkyCC/data/ARC-pred/pred_output/JMA_MSM/' \
                   '%04d/%02d/%02d/%02d/vis/' % (USER_DIR, year, month, day, bt)

        if not os.path.exists('%s/%s.csv' % (save_dir, data_name)):
            get_predict_data(url_pred, save_dir)

        '''
        url_metar = 'http://imaging1.wni.co.jp/SKY/view_metaf.cgi?sty=&fcst=&mode=METAR&area=13'
        dict_metar = get_metar_area(url_metar)
    
        url_metar2 = 'http://imaging1.wni.co.jp/SKY/view_metaf.cgi?sty=&fcst=&mode=METAR&area=14'
        dict_metar.update(get_metar_area(url_metar2))
    
        url_metar3 = 'http://imaging1.wni.co.jp/SKY/view_metaf.cgi?sty=&fcst=&mode=METAR&area=15'
        dict_metar.update(get_metar_area(url_metar3))
    
        url_metar4 = 'http://imaging1.wni.co.jp/SKY/view_metaf.cgi?sty=&fcst=&mode=METAR&area=16'
        dict_metar.update(get_metar_area(url_metar4))
        '''

        url_metar = 'http://imaging1.wni.co.jp/SKY/view_metaf.cgi?mode=SEQ&sty=1&point=%s&head=on&fcst=' % icao
        metar_latest = get_metar_point(url_metar)
        date = _d2h2m2z_to_y4m2d2h2d2(metar_latest['date'])
        metar_latest.loc[:, 'date'] = date

        metar_dir = '/%s/PycharmProjects/SkyCC/data/evaluate/metar/%04d/%02d/%02d/%02d' \
                    % (USER_DIR, year, month, day, bt)

        if os.path.exists('%s/%s.csv' % (metar_dir, icao)):
            metar = pd.read_csv('%s/%s.csv' % (metar_dir, icao))
            metar = pd.concat([metar, metar_latest])
            metar.drop_duplicates('date', inplace=True)
        else:
            os.makedirs(metar_dir, exist_ok=True)
            metar = metar_latest

        metar.index = metar['date']
        metar.sort_index(inplace=True)
        metar.to_csv('%s/%s.csv' % (metar_dir, icao), index=False)


if __name__ == '__main__':
    main()
