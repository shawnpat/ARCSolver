import numpy as np
from random import randint
import json
import os


class NumpyArrayEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def check_zero_rows_cols(permuted_array, sub_grid_x_dim, sub_grid_y_dim):
    reshaped_array = permuted_array.reshape((sub_grid_x_dim, sub_grid_y_dim))
    return np.any(np.all(reshaped_array == 0, axis=0)) or np.any(
        np.all(reshaped_array == 0, axis=1)
    )


# grid_x_dim = 8
# grid_y_dim = 8
num_tasks = 1
task_count = 0
loop_count = 0

while task_count < num_tasks:
    loop_count += 1
    if loop_count % 1000 == 0:
        print("\nloop_count", loop_count)
        print("task_count", task_count)

    input_grids = []
    output_grids = []

    rand_rot_or_flip = randint(0, 5)
    rand_scale_or_shrink = randint(0, 100)

    # Need 4 inputs, 4 outputs (3 train pairs, 1 test pair)
    pair_count = 0
    while pair_count < 4:
        grid_x_dim = randint(6, 8)
        grid_y_dim = randint(6, 8)

        sub_grid_x_dim = randint(2, 4)
        sub_grid_y_dim = randint(2, 4)
        permuted_array = np.random.randint(10, size=sub_grid_x_dim * sub_grid_y_dim)

        if check_zero_rows_cols(permuted_array, sub_grid_x_dim, sub_grid_y_dim):
            continue

        sub_grid = permuted_array.reshape((sub_grid_x_dim, sub_grid_y_dim))

        isolated_obj_grid = np.zeros((grid_x_dim, grid_y_dim), dtype=int)

        start_x_index = randint(0, ((grid_x_dim - (sub_grid_x_dim * 2)) // 2))
        start_y_index = randint(0, ((grid_y_dim - (sub_grid_y_dim * 2)) // 2))

        isolated_obj_grid[
            start_x_index : start_x_index + sub_grid_x_dim,
            start_y_index : start_y_index + sub_grid_y_dim,
        ] = sub_grid

        input_grid = np.copy(isolated_obj_grid)

        output_grid = np.zeros((grid_x_dim, grid_y_dim), dtype=int)
        scaled_sub_grid = np.copy(sub_grid)
        scaled_sub_grid_x_dim = 2 * sub_grid_x_dim
        scaled_sub_grid_y_dim = 2 * sub_grid_y_dim
        # Scaling up the original array
        k = 2
        scaled_sub_grid = np.kron(scaled_sub_grid, np.ones((k, k)))

        output_grid[
            start_x_index : start_x_index + scaled_sub_grid_x_dim,
            start_y_index : start_y_index + scaled_sub_grid_y_dim,
        ] = scaled_sub_grid

        if rand_rot_or_flip > 0 and rand_rot_or_flip < 4:
            input_grid = np.rot90(input_grid, k=rand_rot_or_flip)
            output_grid = np.rot90(output_grid, k=rand_rot_or_flip)
        elif rand_rot_or_flip == 4:
            input_grid = np.fliplr(input_grid)
            output_grid = np.fliplr(output_grid)
        elif rand_rot_or_flip == 5:
            input_grid = np.flipud(input_grid)
            output_grid = np.flipud(output_grid)

        if rand_scale_or_shrink > 50:
            input_grids.append(output_grid)
            output_grids.append(input_grid)
        else:
            input_grids.append(input_grid)
            output_grids.append(output_grid)

        pair_count += 1

    task = {
        "train": [
            {"input": [], "output": []},
            {"input": [], "output": []},
            {"input": [], "output": []},
        ],
        "test": [{"input": [], "output": []}],
    }

    for i, (input_grid, output_grid) in enumerate(zip(input_grids, output_grids)):
        # for input_grid, output_grid in zip(input_grids, output_grids):
        if i < 3:
            task["train"][i]["input"] = input_grid
            task["train"][i]["output"] = output_grid
        else:
            task["test"][0]["input"] = input_grid
            task["test"][0]["output"] = output_grid
        i += 1

    if num_tasks > 1:
        json_string = json.dumps(task, cls=NumpyArrayEncoder)

        data_folder = "data/"

        data_name = "scale_obj"
        filename = data_name + "_" + str(task_count) + ".json"

        train_or_test_folders = ["train", "test"]
        random_int = randint(1, 100)
        if random_int <= 80:
            train_or_test_folder = train_or_test_folders[0]
        else:
            train_or_test_folder = train_or_test_folders[1]

        folder_path = data_folder + train_or_test_folder + "/"

        full_folder_path = folder_path + data_name
        if not os.path.exists(full_folder_path):
            os.mkdir(full_folder_path)

        filepath = full_folder_path + "/" + filename
        with open(filepath, "w") as outfile:
            outfile.write(json_string)
    else:
        print(task)

    task_count += 1
