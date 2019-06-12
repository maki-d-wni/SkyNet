import pickle
import numpy as np
import pandas as pd

from SkyCC.error_analysis.base import *


def main():
    from skynet.data_handling import split_time_series
    data_path = "/Users/makino/PycharmProjects/SkyCC/data/pickle/learning/svm"

    icao = "RJFT"
    data = pickle.load(open("%s/%s" % (data_path, "test_%s.pkl" % icao), "rb"))

    # area forecast
    af = data[["date", "VIS"]]
    af.columns = ["date", "visibility"]

    spaf = split_time_series(af, level="month", period=2)

    # human edit
    he = set_visibility_human_edit(icao)
    sphe = split_time_series(he, level="month", period=2)

    # 期間別VIS

    for key in spaf:
        x_af = copy.deepcopy(spaf[key])
        x_af.index = x_af["date"].astype(int).astype(str).values

        x_he = copy.deepcopy(sphe[key])
        x_he.index = x_he["date"].astype(int).astype(str).values

        x = copy.deepcopy(x_he[["date"]])
        x["AF_visibility"] = x_af["visibility"]
        x["human_visibility"] = x_he["visibility"]
        x = x.dropna()

        """
        make_animation2(x, act="metar", pred="AF", act_color="b", pred_color="r",
                        ani_name=os.getcwd() + "/movie/vis_metar_AF_%s_%s.mp4" % (icao, key))
        """

        make_animation2(x, act="human", pred="AF", act_color="g", pred_color="r",
                        ani_name=os.getcwd() + "/movie/vis_human_AF_%s_%s.mp4" % (icao, key))


if __name__ == "__main__":
    main()
