import os
import numpy as np
from progiter import ProgIter
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, ConcatDataset, random_split
import os
import json
import math
import copy
from typing import Optional, Any, Union, Callable
from torch import Tensor
from torch.nn.modules.module import Module
from torch.nn.modules.activation import MultiheadAttention
from torch.nn.modules.container import ModuleList
from torch.nn.init import xavier_uniform_
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.linear import Linear
from torch.nn.modules.normalization import LayerNorm
from torch.nn.modules.conv import Conv1d
import shutil

root_dir = "data/ARC/"
file_list = []

max_train_tasks = 0
max_test_tasks = 0

min_train_tasks = 100
min_test_tasks = 100

num_of_1_train_puzzles = 0
num_of_2_train_puzzles = 0
num_of_3_train_puzzles = 0
num_of_4_train_puzzles = 0
num_of_5_train_puzzles = 0
num_of_6_train_puzzles = 0
num_of_7_train_puzzles = 0
num_of_8_train_puzzles = 0
num_of_9_train_puzzles = 0
num_of_10_train_puzzles = 0

num_of_1_test_puzzles = 0
num_of_2_test_puzzles = 0
num_of_3_test_puzzles = 0

max_grid_squared_dim = 512

for root, dirs, files in os.walk(root_dir):
    for name in files:
        file_path = os.path.join(root, name)
        file_list.append(file_path)

puzzle_count = 0
for idx, fp in enumerate(file_list):
    file_path = file_list[idx]

    with open(file_path, "r") as f:
        data = json.load(f)

    train_tasks = data["train"]
    test_tasks = data["test"]

    too_big = False
    for task in train_tasks:
        if len(task["input"]) > max_grid_squared_dim:
            too_big = True
            break
    if too_big:
        continue

    train_len = len(train_tasks)

    if train_len != 3:
        continue

    # if train_len == 1:
    #     num_of_1_train_puzzles += 1
    # elif train_len == 2:
    #     num_of_2_train_puzzles += 1
    # elif train_len == 3:
    #     num_of_3_train_puzzles += 1
    # elif train_len == 4:
    #     num_of_4_train_puzzles += 1
    # elif train_len == 5:
    #     num_of_5_train_puzzles += 1
    # elif train_len == 6:
    #     num_of_6_train_puzzles += 1
    # elif train_len == 7:
    #     num_of_7_train_puzzles += 1
    # elif train_len == 8:
    #     num_of_8_train_puzzles += 1
    # elif train_len == 9:
    #     num_of_9_train_puzzles += 1
    # elif train_len == 10:
    #     num_of_10_train_puzzles += 1

    # if train_len > max_train_tasks:
    #     max_train_tasks = train_len
    # if train_len < min_train_tasks:
    #     min_train_tasks = train_len

    test_len = len(test_tasks)
    if test_len == 1:
        num_of_1_test_puzzles += 1
    elif test_len == 2:
        num_of_2_test_puzzles += 1
    elif test_len == 3:
        num_of_3_test_puzzles += 1

    puzzle_count += 1

    # if test_len > max_test_tasks:
    #     max_test_tasks = test_len
    # if test_len < min_test_tasks:
    #     min_test_tasks = test_len

    # for i, train_task in enumerate(train_tasks):
    #     input_grid = np.array(train_task["input"])
    #     output_grid = np.array(train_task["output"])
    # print(
    #     "input_grid.shape",
    #     input_grid.shape,
    #     "  output_grid.shape",
    #     output_grid.shape,
    # )

    # for i, test_task in enumerate(test_tasks):
    #     input_grid = np.array(test_task["input"])
    #     output_grid = np.array(train_task["output"])
    # print(
    #     "input_grid.shape",
    #     input_grid.shape,
    #     "  output_grid.shape",
    #     output_grid.shape,
    # )

print(puzzle_count)

# print("min_train_tasks", min_train_tasks)
# print("max_train_tasks", max_train_tasks)
# print("min_test_tasks", min_test_tasks)
# print("max_test_tasks", max_test_tasks)

# min_train_tasks 2
# max_train_tasks 10

# min_test_tasks 1
# max_test_tasks 3

# max grid size = 30x30

# print("num_of_1_train_puzzles", num_of_1_train_puzzles)
# print("num_of_2_train_puzzles", num_of_2_train_puzzles)
# print("num_of_3_train_puzzles", num_of_3_train_puzzles)
# print("num_of_4_train_puzzles", num_of_4_train_puzzles)
# print("num_of_5_train_puzzles", num_of_5_train_puzzles)
# print("num_of_6_train_puzzles", num_of_6_train_puzzles)
# print("num_of_7_train_puzzles", num_of_7_train_puzzles)
# print("num_of_8_train_puzzles", num_of_8_train_puzzles)
# print("num_of_9_train_puzzles", num_of_9_train_puzzles)
# print("num_of_10_train_puzzles", num_of_10_train_puzzles)

# print("num_of_1_test_puzzles", num_of_1_test_puzzles)
# print("num_of_2_test_puzzles", num_of_2_test_puzzles)
# print("num_of_3_test_puzzles", num_of_3_test_puzzles)

# num_of_1_train_puzzles 0
# num_of_2_train_puzzles 102
# num_of_3_train_puzzles 453
# num_of_4_train_puzzles 168
# num_of_5_train_puzzles 48
# num_of_6_train_puzzles 19
# num_of_7_train_puzzles 7
# num_of_8_train_puzzles 2
# num_of_9_train_puzzles 0
# num_of_10_train_puzzles 1

# num_of_1_test_puzzles 767
# num_of_2_test_puzzles 31
# num_of_3_test_puzzles 2
