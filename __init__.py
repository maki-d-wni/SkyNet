import json

__config = json.load(open("/Users/makino/PycharmProjects/skyer/config.json"))
DATA_PATH = __config["path"]["data"]
SKYNET_PATH = __config["path"]["skynet"]
IMAGE_PATH = __config["path"]["image"]
MOVIE_PATH = __config["path"]["movie"]
LIVE_PATH = __config["path"]["live"]
MODEL_PATH = __config["path"]["trained_model"]
