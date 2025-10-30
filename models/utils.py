#========== 工具函数 ==========
import os
import json

def create_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)

def get_index1(lst, item):
    return [idx for idx, val in enumerate(lst) if val == item]

def load_json(file_path):
    with open(file_path, "r") as f:
        return json.load(f)