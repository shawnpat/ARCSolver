import json
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import random


def plot_task(task):
    """
    Plots all train and test pairs of a specified task,
    using same color scheme as the ARC app
    """
    # cmap = plt.cm.tab10
    cmap = matplotlib.colors.ListedColormap(
        [
            "#000000",
            "#0074D9",
            "#FF4136",
            "#2ECC40",
            "#FFDC00",
            "#AAAAAA",
            "#F012BE",
            "#FF851B",
            "#7FDBFF",
            "#870C25",
        ]
    )
    norm = plt.Normalize(vmin=0, vmax=9)

    n_train = len(task["train"])
    n_test = len(task["test"])

    fig, axs = plt.subplots(n_train + n_test, 2, figsize=(4, 2 * (n_train + n_test)))

    for i in range(n_train):
        axs[i, 0].imshow(task["train"][i]["input"], cmap=cmap, norm=norm)
        axs[i, 0].axis("off")
        axs[i, 0].set_title("Train Input {}".format(i + 1))
        axs[i, 1].imshow(task["train"][i]["output"], cmap=cmap, norm=norm)
        axs[i, 1].axis("off")
        axs[i, 1].set_title("Train Output {}".format(i + 1))

    for i in range(n_test):
        axs[i + n_train, 0].imshow(task["test"][i]["input"], cmap=cmap, norm=norm)
        axs[i + n_train, 0].axis("off")
        axs[i + n_train, 0].set_title("Test Input {}".format(i + 1))
        axs[i + n_train, 1].imshow(task["test"][i]["output"], cmap=cmap, norm=norm)
        axs[i + n_train, 1].axis("off")
        axs[i + n_train, 1].set_title("Test Output {}".format(i + 1))

    # plt.tight_layout()
    plt.show()


def open_arc_json(json_file):
    with open(json_file, "r") as f:
        task = json.load(f)
    return task


def pick_random_file(directory):
    subdirs = [
        d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))
    ]
    if not subdirs:
        json_files = [f for f in os.listdir(directory) if f.endswith(".json")]
        if json_files:
            index = random.randint(0, len(json_files) - 1)
            return os.path.join(directory, json_files[index])
    subdir = os.path.join(directory, random.choice(subdirs))
    for root, dirs, files in os.walk(subdir):
        json_files = [f for f in files if f.endswith(".json")]
        if json_files:
            index = random.randint(0, len(json_files) - 1)
            return os.path.join(root, json_files[index])
    return None


directory = "data/ARC/training/"
file_path = ""
while not file_path.endswith(".json"):
    file_path = pick_random_file(directory)
# file_path = "data/temp.json"
print(file_path)
task = open_arc_json(file_path)
plot_task(task)
