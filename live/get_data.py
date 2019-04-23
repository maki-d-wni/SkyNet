import os
import re
import requests
from bs4 import BeautifulSoup


def get_predict_data(url, save_dir='./'):
    os.makedirs(save_dir, exist_ok=True)
    os.system('wget %s -P %s' % (url, save_dir))


def get_metar(url):
    url_info = requests.get(url)
    html = url_info.content

    soup = BeautifulSoup(html, 'html.parser')

    tbl = soup.find_all('td')
    for t in tbl:
        pattern = re.compile('<td>RJ.+<br/>')
        r = pattern.match(str(t))
        if r is not None:
            metar = r.group()
            print(metar)


def main():
    from skynet import DATA_DIR

    year = 2019
    month = 4
    day = 18
    bt = 0
    icao = 'RJAA'
    data_name = 'GLOBAL_METAR-%s.vis' % icao

    url_pred = 'http://pt-compass-arc.wni.co.jp/compass/data/ARC-pred/pred_output/JMA_MSM/' \
               '%04d/%02d/%02d/%02d/vis/%s.csv' % (year, month, day, bt, data_name)

    save_dir = '/Users/makino/PycharmProjects/SkyCC/data/ARC-pred/pred_output/JMA_MSM/' \
               '%04d/%02d/%02d/%02d/vis/' % (year, month, day, bt)

    # get_predict_data(url_pred, save_dir)

    url_metar = 'http://imaging1.wni.co.jp/SKY/view_metaf.cgi?sty=&fcst=&mode=METAR&area=15'
    get_metar(url_metar)
    

if __name__ == '__main__':
    main()
