import numpy as np
import torch
import os
import json

# THIS ONE DOES NOT USE THE TEST OUTPUTS


np.random.seed(0)
torch.manual_seed(0)

# Defining model and training options
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(
    "Using device: ",
    device,
    f"({torch.cuda.get_device_name(device)})" if torch.cuda.is_available() else "",
)


class NumpyArrayEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def main():
    rot_range = 4
    flip_range = 2
    color_range = 9

    puzzle_num = 0
    root_dir = "data/step_2/evaluation_max_3_trains_1_test/"
    file_list = os.listdir(root_dir)
    for file in file_list:
        puzzle_num += 1

        print(puzzle_num, ". file: ", file)
        file_path = os.path.join(root_dir, file)

        with open(file_path, "r") as f:
            data = json.load(f)

        train_tasks = data["train"]
        test_tasks = data["test"]
        num_train_tasks = len(train_tasks)
        num_test_tasks = len(test_tasks)

        ################### START AUGMENTATIONS ##################

        aug_range = num_train_tasks
        num_augmented = rot_range * flip_range * color_range * aug_range
        print("num_augmented", num_augmented)

        colors = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
        shifted_colors = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])

        augmentation_count = 0
        for rot_num in range(rot_range):
            for flip_num in range(flip_range):
                for color_num in range(color_range):
                    for aug_num in range(aug_range):
                        augmentation_count += 1

                        task = {
                            "train": [
                                {"input": [], "output": []},
                            ],
                            "test": [{"input": [], "output": []}],
                        }

                        while len(task["train"]) < num_train_tasks - 1:
                            task["train"].append({"input": [], "output": []})

                        while len(task["test"]) < num_test_tasks:
                            task["test"].append({"input": [], "output": []})

                        augmented_test_tasks = [train_tasks[aug_num]]

                        next_aug = aug_num + 1
                        if next_aug > num_train_tasks - 1:
                            next_aug = next_aug - num_train_tasks

                        augmented_train_tasks = [
                            train_tasks[next_aug],
                        ]

                        while len(augmented_train_tasks) < num_train_tasks - 1:
                            next_aug += 1
                            if next_aug > num_train_tasks - 1:
                                next_aug = next_aug - num_train_tasks
                            augmented_train_tasks.append(train_tasks[next_aug])

                        for i, augmented_train_task in enumerate(augmented_train_tasks):
                            input = np.array(augmented_train_task["input"])
                            output = np.array(augmented_train_task["output"])

                            input_grid_dim_x = input.shape[0]
                            input_grid_dim_y = input.shape[1]

                            output_grid_dim_x = output.shape[0]
                            output_grid_dim_y = output.shape[1]

                            if color_num > 0:
                                for orig_index in range(len(colors)):
                                    swap_index = orig_index + color_num
                                    if swap_index >= len(colors):
                                        swap_index = (orig_index + color_num) - len(
                                            colors
                                        )
                                    shifted_colors[orig_index] = colors[swap_index]
                                mapping = dict(zip(colors, shifted_colors))

                                for x in range(input_grid_dim_x):
                                    for y in range(input_grid_dim_y):
                                        if input[x, y] != 0:
                                            input[x, y] = mapping[input[x, y]]

                                for x in range(output_grid_dim_x):
                                    for y in range(output_grid_dim_y):
                                        if output[x, y] != 0:
                                            output[x, y] = mapping[output[x, y]]

                            if rot_num > 0:
                                input = np.rot90(input, k=rot_num).copy()
                                output = np.rot90(output, k=rot_num).copy()

                            if flip_num > 0:
                                if flip_num == 1:
                                    input = np.flipud(input).copy()
                                    output = np.flipud(output).copy()
                                else:
                                    input = np.fliplr(input).copy()
                                    output = np.fliplr(output).copy()

                            task["train"][i]["input"] = input
                            task["train"][i]["output"] = output

                        for i, augmented_test_task in enumerate(augmented_test_tasks):
                            input = np.array(augmented_test_task["input"])
                            output = np.array(augmented_test_task["output"])

                            input_grid_dim_x = input.shape[0]
                            input_grid_dim_y = input.shape[1]

                            output_grid_dim_x = output.shape[0]
                            output_grid_dim_y = output.shape[1]

                            if color_num > 0:
                                for orig_index in range(len(colors)):
                                    swap_index = orig_index + color_num
                                    if swap_index >= len(colors):
                                        swap_index = (orig_index + color_num) - len(
                                            colors
                                        )
                                    shifted_colors[orig_index] = colors[swap_index]
                                mapping = dict(zip(colors, shifted_colors))

                                for x in range(input_grid_dim_x):
                                    for y in range(input_grid_dim_y):
                                        if input[x, y] != 0:
                                            input[x, y] = mapping[input[x, y]]

                                for x in range(output_grid_dim_x):
                                    for y in range(output_grid_dim_y):
                                        if output[x, y] != 0:
                                            output[x, y] = mapping[output[x, y]]

                            if rot_num > 0:
                                input = np.rot90(input, k=rot_num).copy()
                                output = np.rot90(output, k=rot_num).copy()

                            if flip_num > 0:
                                if flip_num == 1:
                                    input = np.flipud(input).copy()
                                    output = np.flipud(output).copy()
                                else:
                                    input = np.fliplr(input).copy()
                                    output = np.fliplr(output).copy()

                            task["test"][i]["input"] = input
                            task["test"][i]["output"] = output

                        json_string = json.dumps(task, cls=NumpyArrayEncoder)
                        data_folder = "data/step_2/evaluation_augs/"
                        base_file_name = file[:-5]
                        augmented_filename = (
                            base_file_name + "_" + str(augmentation_count) + ".json"
                        )
                        if not os.path.exists(data_folder):
                            os.mkdir(data_folder)

                        filepath = data_folder + augmented_filename
                        with open(filepath, "w") as outfile:
                            outfile.write(json_string)
                            # print("Saved: ", filepath)


if __name__ == "__main__":
    main()
