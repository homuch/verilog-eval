import json

_MODEL_PATH_DICT_PATH = "model-path-dict.json"

def load_model_path_dict():
    with open(_MODEL_PATH_DICT_PATH, 'r') as f:
        return json.load(f)

MODEL_PATH_DICT = load_model_path_dict()

BUILD_PATH = "/build"
DATASET_PATH_PREFIX = "/dataset_"
