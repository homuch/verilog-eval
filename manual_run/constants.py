import json
import os
script_dir = os.path.dirname(os.path.abspath(__file__))

_MODEL_PATH_DICT_PATH = os.path.join(script_dir, "model-path-dict.json")

def load_model_path_dict():
    with open(_MODEL_PATH_DICT_PATH, 'r') as f:
        return json.load(f)

MODEL_PATH_DICT = load_model_path_dict()

BUILD_PATH = os.path.join(script_dir, "../build")
DATASET_PATH_PREFIX = os.path.join(script_dir, "../dataset_")
