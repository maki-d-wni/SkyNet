import os
import pandas as pd
import matplotlib.pyplot as plt


def main():
    # icaos = ["RJFT", "RJFK", "RJCC"]
    icaos = ['RJOT']
    os.makedirs(os.getcwd() + "/graph", exist_ok=True)
    for icao in icaos:
        score = pd.read_html(os.getcwd() + "/score/%s.html" % icao)[0]

        x = score["confidence"].values
        plt.figure()
        plt.plot(x, score["%"].values / 100, linewidth=3., label="reduction rate of edit")
        plt.plot(x, score["f1"].values, linewidth=3, label="f1")
        plt.plot(x, score["recall"].values, linewidth=3., label="recall")
        plt.plot(x, score["precision"].values, linewidth=3., label="precision")
        plt.title(icao, fontsize=12)
        plt.xlabel("confidence (%)", fontsize=12)
        plt.legend()
        plt.ylim(0, 1)
        plt.grid(linestyle="--")
        plt.savefig(os.getcwd() + "/graph/%s.png" % icao)


if __name__ == "__main__":
    main()
