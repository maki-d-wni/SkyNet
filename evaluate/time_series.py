import os
import pandas as pd
import matplotlib.pyplot as plt


def main():
    # icaos = ["RJFT", "RJFK", "RJCC"]
    icaos = ['RJFK']
    month_keys = ['month:1-2', 'month:3-4', 'month:5-6', 'month:7-8', 'month:9-10', 'month:11-12']
    confidence = range(10, 100, 10)
    vis_class = range(9)
    edit_keys = ['edit class %s' % i for i in vis_class]
    os.makedirs(os.getcwd() + "/graph", exist_ok=True)
    for icao in icaos:
        score = pd.DataFrame(
            index=month_keys, columns=edit_keys
        )
        for month_key in month_keys:
            for c in confidence[0:1]:
                vis = pd.read_html(
                    os.getcwd() + "/confidence_factor/%s/%s/%s_%s.html"
                    % (month_key, icao, icao, c)
                )[0].iloc[:, 1:]

                all_samples = len(vis)

                vis.dropna(inplace=True)

                p_ml = vis['skynet'].values
                p_metar = vis['metar'].values
                p_he = vis['human'].values

                score.loc[month_key, edit_keys] = [len(p_ml[p_ml == cls]) for cls in vis_class]
                score.loc[month_key, 'all edit samples'] = len(p_ml)
                score.loc[month_key, 'all samples'] = all_samples

        print(score)


if __name__ == "__main__":
    main()
