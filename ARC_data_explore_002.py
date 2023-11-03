import os
import numpy as np
from progiter import ProgIter
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, ConcatDataset, random_split
import os
import json


root_dir = "data/ARC/"
file_list = []

max_train_tasks = 0
max_test_tasks = 0

min_train_tasks = 100
min_test_tasks = 100

max_allowed_grid_dim = 22

for root, dirs, files in os.walk(root_dir):
    for name in files:
        file_path = os.path.join(root, name)
        file_list.append(file_path)

puzzle_count = 0
multi_test_count = 0
for idx, fp in enumerate(file_list):
    file_path = file_list[idx]

    with open(file_path, "r") as f:
        data = json.load(f)

    train_tasks = data["train"]
    test_tasks = data["test"]

    too_big = False
    for task in train_tasks:
        if (
            len(task["input"]) > max_allowed_grid_dim
            or len(task["input"][0]) > max_allowed_grid_dim
        ):
            too_big = True
            break
    if too_big:
        continue

    train_len = len(train_tasks)

    if train_len != 3:
        continue

    test_len = len(test_tasks)
    if test_len > 1:
        multi_test_count += 1

    puzzle_count += 1

print("puzzle_count", puzzle_count)
print("multi_test_count", multi_test_count)
