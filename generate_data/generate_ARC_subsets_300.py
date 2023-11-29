from email.mime import base
import numpy as np
import os
import json


class NumpyArrayEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def make_json_file(base_file_name, train_tasks, test_tasks):
    data = {"train": train_tasks, "test": test_tasks}
    json_string = json.dumps(data, cls=NumpyArrayEncoder)
    data_folder = "data/step_2/evaluation_max_3_trains_1_test/"

    augmented_filename = base_file_name + ".json"
    if not os.path.exists(data_folder):
        os.mkdir(data_folder)

    filepath = data_folder + augmented_filename
    with open(filepath, "w") as outfile:
        outfile.write(json_string)


def main():
    root_dir = "data/ARC/evaluation/"
    for root, dirs, files in os.walk(root_dir):
        for name in files:
            if ".json" not in name:
                continue

            file_path = os.path.join(root, name)
            print("file_path", file_path)

            base_file_name = name[:-6]
            print("base_file_name", base_file_name)

            with open(file_path, "r") as f:
                data = json.load(f)

            train_tasks = data["train"]
            test_tasks = data["test"]

            if len(train_tasks) > 3:
                train_tasks = train_tasks[:3]

            for i, test_task in enumerate(test_tasks):
                if i > 0:
                    base_file_name = base_file_name + "_" + str(i + 1)
                test_task = [test_task]
                make_json_file(base_file_name, train_tasks, test_task)


if __name__ == "__main__":
    main()
    print("Done!")
