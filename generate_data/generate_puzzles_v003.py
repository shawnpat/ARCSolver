from platform import win32_edition
import numpy as np
import random
from random import randint
import json
import os

# import matplotlib
# import matplotlib.pyplot as plt


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


def generate_count_squares_puzzle(
    num_train_pairs, num_test_pairs, min_grid_dim, max_grid_dim
):
    input_grids = []
    output_grids = []

    colors = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    rot_num = randint(0, 3)
    flip_num = randint(0, 1)

    pair_count = 0
    while pair_count < num_train_pairs + num_test_pairs:
        grid_x_dim = randint(min_grid_dim, max_grid_dim)
        grid_y_dim = randint(min_grid_dim, max_grid_dim)

        color_1 = colors[randint(1, 9)]
        color_2 = colors[randint(1, 9)]
        while color_2 == color_1:
            color_2 = colors[randint(1, 9)]
        final_num_color_1 = randint(2, 32)
        final_num_color_2 = randint(1, final_num_color_1 - 1)
        num_color_1 = 0
        num_color_2 = 0

        input_grid = np.zeros((grid_x_dim, grid_y_dim), dtype=int)
        w2_loop_count = 0
        while num_color_1 < final_num_color_1:
            w2_loop_count += 1
            if w2_loop_count > 1000:
                return ""
            random_x = randint(0, grid_x_dim - 1)
            random_y = randint(0, grid_y_dim - 1)
            if input_grid[random_x][random_y] == 0:
                input_grid[random_x][random_y] = color_1
                num_color_1 += 1
        w3_loop_count = 0
        while num_color_2 < final_num_color_2:
            w3_loop_count += 1
            if w3_loop_count > 1000:
                return ""
            random_x = randint(0, grid_x_dim - 1)
            random_y = randint(0, grid_y_dim - 1)
            if input_grid[random_x][random_y] == 0:
                input_grid[random_x][random_y] = color_2
                num_color_2 += 1

        input_grids.append(input_grid)

        output_grid = np.zeros((grid_x_dim, grid_y_dim), dtype=int)
        num_color_1 = 0
        num_color_2 = 0
        for x in range(grid_x_dim):
            for y in range(grid_y_dim):
                if num_color_1 < final_num_color_1:
                    output_grid[x][y] = color_1
                    num_color_1 += 1
                elif num_color_2 < final_num_color_2:
                    output_grid[x][y] = color_2
                    num_color_2 += 1
        if flip_num > 0:
            output_grid = np.flipud(output_grid)
        if rot_num > 0:
            output_grid = np.rot90(output_grid, k=rot_num)

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
        if i < num_train_pairs:
            task["train"][i]["input"] = input_grid
            task["train"][i]["output"] = output_grid
        else:
            task["test"][0]["input"] = input_grid
            task["test"][0]["output"] = output_grid
        i += 1

    json_string = json.dumps(task, cls=NumpyArrayEncoder)
    return json_string


def generate_fill_holes_in_line_patterns_puzzle(
    num_train_pairs, num_test_pairs, min_grid_dim, max_grid_dim
):
    grid_x_dim = randint(min_grid_dim, max_grid_dim)
    grid_y_dim = randint(min_grid_dim, max_grid_dim)

    input_grids = []
    output_grids = []

    short_line_patterns = [
        [0],
        [0, 1],
        [0, 0, 1],
        [0, 0, 1, 1],
        [0, 0, 0, 1],
        [0, 1, 2],
        [0, 1, 2, 3],
        [0, 1, 2, 3, 4],
        [0, 0, 1, 1, 2, 2],
    ]

    colors = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    rot = random.choice([True, False])
    flip = random.choice([True, False])

    pair_count = 0
    w_loop_count = 0
    while pair_count < num_train_pairs + num_test_pairs:
        w_loop_count += 1
        if w_loop_count > 1000:
            return ""
        output_grid = np.zeros((grid_x_dim, grid_y_dim), dtype=int)
        # Define the short numpy array
        random_index = randint(0, len(short_line_patterns) - 1)  # 0, 1, 2, 3, 4
        short_array = np.array(short_line_patterns[random_index])

        # Define the length of the desired 1D numpy array
        length = grid_y_dim

        # Create the 1D numpy array by repeating the short array
        long_array = np.tile(short_array, length // len(short_array) + 1)[:length]

        for row_num in range(grid_x_dim):
            color0 = colors[randint(1, 9)]

            color1 = colors[randint(1, 9)]
            while color1 == color0:
                color1 = colors[randint(1, 9)]

            color2 = colors[randint(1, 9)]
            while color2 == color1 or color2 == color0:
                color2 = colors[randint(1, 9)]

            color3 = colors[randint(1, 9)]
            while color3 == color2 or color3 == color1 or color3 == color0:
                color3 = colors[randint(1, 9)]

            color4 = colors[randint(1, 9)]
            while (
                color4 == color3
                or color4 == color2
                or color4 == color1
                or color4 == color0
            ):
                color4 = colors[randint(1, 9)]

            line_pattern_copy = np.copy(long_array)
            for i in range(len(line_pattern_copy)):
                if line_pattern_copy[i] == 0:
                    line_pattern_copy[i] = color0
                elif line_pattern_copy[i] == 1:
                    line_pattern_copy[i] = color1
                elif line_pattern_copy[i] == 2:
                    line_pattern_copy[i] = color2
                elif line_pattern_copy[i] == 3:
                    line_pattern_copy[i] = color3
                elif line_pattern_copy[i] == 4:
                    line_pattern_copy[i] = color4

            output_grid[
                row_num,
                :,
            ] = line_pattern_copy

        input_grid = np.copy(output_grid)
        for y in range(grid_y_dim):
            for x in range(grid_x_dim):
                make_black = randint(1, 100)
                if make_black > 90:
                    input_grid[x][y] = 0

        if flip:
            input_grid = np.fliplr(input_grid)
            output_grid = np.fliplr(output_grid)
        if rot:
            input_grid = np.rot90(input_grid, k=1)
            output_grid = np.rot90(output_grid, k=1)
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
        if i < 3:
            task["train"][i]["input"] = input_grid
            task["train"][i]["output"] = output_grid
        else:
            task["test"][0]["input"] = input_grid
            task["test"][0]["output"] = output_grid
        i += 1

    json_string = json.dumps(task, cls=NumpyArrayEncoder)

    return json_string


def generate_fill_holes_in_pattern_puzzle_v1(
    num_train_pairs, num_test_pairs, min_grid_dim, max_grid_dim
):
    input_grids = []
    output_grids = []

    pair_count = 0
    w_loop_count = 0
    while pair_count < num_train_pairs + num_test_pairs:
        w_loop_count += 1
        if w_loop_count > 1000:
            return ""
        grid_x_dim = randint(min_grid_dim, max_grid_dim)
        while grid_x_dim % 2 > 0:
            grid_x_dim = randint(min_grid_dim, max_grid_dim)
        grid_y_dim = randint(min_grid_dim, max_grid_dim)
        while grid_y_dim % 2 > 0:
            grid_y_dim = randint(min_grid_dim, max_grid_dim)

        quarter_grid_dim_x = grid_x_dim // 2
        quarter_grid_dim_y = grid_y_dim // 2

        output_grid = np.zeros((grid_x_dim, grid_y_dim), dtype=int)
        flat_quarter_grid = np.random.randint(
            1, 10, size=(quarter_grid_dim_x * quarter_grid_dim_y)
        )
        quarter_1_grid = flat_quarter_grid.reshape(
            (quarter_grid_dim_x, quarter_grid_dim_y)
        )

        output_grid[
            0:quarter_grid_dim_x,
            0:quarter_grid_dim_y,
        ] = quarter_1_grid

        output_grid[
            quarter_grid_dim_x:,
            0:quarter_grid_dim_y,
        ] = np.flipud(quarter_1_grid)

        output_grid[
            0:quarter_grid_dim_x,
            quarter_grid_dim_y:,
        ] = np.fliplr(quarter_1_grid)

        output_grid[
            quarter_grid_dim_x:,
            quarter_grid_dim_y:,
        ] = np.fliplr(np.flipud(quarter_1_grid))

        output_grids.append(output_grid)

        input_grid = np.copy(output_grid)
        no_black_squares = True
        w2_loop_count = 0
        while no_black_squares:
            w2_loop_count += 1
            if w2_loop_count > 1000:
                return ""
            for x in range(grid_x_dim):
                for y in range(grid_y_dim):
                    make_black = randint(1, 100)
                    if make_black > 95:
                        input_grid[x][y] = 0
                        no_black_squares = False
        input_grids.append(input_grid)

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

    json_string = json.dumps(task, cls=NumpyArrayEncoder)

    return json_string


def generate_fill_holes_in_pattern_puzzle_v2(
    num_train_pairs, num_test_pairs, min_grid_dim, max_grid_dim
):
    input_grids = []
    output_grids = []

    flip = random.choice([True, False])
    pair_count = 0
    w_loop_count = 0
    while pair_count < num_train_pairs + num_test_pairs:
        w_loop_count += 1
        if w_loop_count > 1000:
            return ""
        grid_x_dim = randint(min_grid_dim, max_grid_dim)
        while grid_x_dim % 2 > 0:
            grid_x_dim = randint(min_grid_dim, max_grid_dim)
        grid_y_dim = grid_x_dim

        quarter_grid_dim_x = grid_x_dim // 2
        quarter_grid_dim_y = grid_y_dim // 2

        output_grid = np.zeros((grid_x_dim, grid_y_dim), dtype=int)
        flat_quarter_grid = np.random.randint(
            1, 10, size=(quarter_grid_dim_x * quarter_grid_dim_y)
        )
        quarter_1_grid = flat_quarter_grid.reshape(
            (quarter_grid_dim_x, quarter_grid_dim_y)
        )

        output_grid[
            0:quarter_grid_dim_x,
            0:quarter_grid_dim_y,
        ] = quarter_1_grid

        if flip:
            output_grid[
                quarter_grid_dim_x:,
                0:quarter_grid_dim_y,
            ] = np.flipud(quarter_1_grid)

            output_grid[
                0:quarter_grid_dim_x,
                quarter_grid_dim_y:,
            ] = np.fliplr(quarter_1_grid)

            output_grid[
                quarter_grid_dim_x:,
                quarter_grid_dim_y:,
            ] = np.fliplr(np.flipud(quarter_1_grid))
        else:
            output_grid[
                quarter_grid_dim_x:,
                0:quarter_grid_dim_y,
            ] = np.rot90(quarter_1_grid, k=1)

            output_grid[
                0:quarter_grid_dim_x,
                quarter_grid_dim_y:,
            ] = np.rot90(quarter_1_grid, k=2)

            output_grid[
                quarter_grid_dim_x:,
                quarter_grid_dim_y:,
            ] = np.rot90(quarter_1_grid, k=3)

        output_grids.append(output_grid)

        input_grid = np.copy(output_grid)
        for x in range(grid_x_dim):
            for y in range(grid_y_dim):
                make_black = randint(1, 100)
                if make_black > 95:
                    input_grid[x][y] = 0
        input_grids.append(input_grid)

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

    json_string = json.dumps(task, cls=NumpyArrayEncoder)

    return json_string


def generate_fill_surrounded_puzzle(
    num_train_pairs, num_test_pairs, min_grid_dim, max_grid_dim
):
    input_grids = []
    output_grids = []

    interior_color = randint(1, 9)
    surround_color = randint(1, 9)
    while surround_color == interior_color:
        surround_color = randint(1, 9)

    fill_exterior = randint(0, 1)

    pair_count = 0
    w_loop_count = 0
    while pair_count < num_train_pairs + num_test_pairs:
        w_loop_count += 1
        if w_loop_count > 1000:
            return ""
        grid_x_dim = randint(min_grid_dim, max_grid_dim)
        grid_y_dim = randint(min_grid_dim, max_grid_dim)

        sub_grid_x_dim = randint(2, 6)
        sub_grid_y_dim = randint(2, 6)

        size = sub_grid_x_dim * sub_grid_y_dim
        permuted_array = np.random.randint(10, size=size)

        if check_zero_rows_cols(permuted_array, sub_grid_x_dim, sub_grid_y_dim):
            continue

        sub_grid = permuted_array.reshape((sub_grid_x_dim, sub_grid_y_dim))

        base_grid = np.zeros((grid_x_dim, grid_y_dim), dtype=int)

        # Place sub-grid randomly inside of base_grid
        rand_start_x_index = randint(1, grid_x_dim - sub_grid_x_dim - 1)
        rand_start_y_index = randint(1, grid_y_dim - sub_grid_y_dim - 1)
        base_grid[
            rand_start_x_index : rand_start_x_index + sub_grid_x_dim,
            rand_start_y_index : rand_start_y_index + sub_grid_y_dim,
        ] = sub_grid

        for x in range(grid_x_dim):
            for y in range(grid_y_dim):
                if base_grid[x][y] != 0:
                    base_grid[x][y] = interior_color

        output_grid = np.copy(base_grid)
        for x in range(grid_x_dim):
            for y in range(grid_y_dim):
                if (
                    output_grid[x][y] == 0
                    and x + 1 < grid_x_dim
                    and output_grid[x + 1][y] == interior_color
                ):
                    output_grid[x][y] = surround_color
                elif (
                    output_grid[x][y] == 0
                    and x - 1 > 0
                    and output_grid[x - 1][y] == interior_color
                ):
                    output_grid[x][y] = surround_color
                elif (
                    output_grid[x][y] == 0
                    and y + 1 < grid_y_dim
                    and output_grid[x][y + 1] == interior_color
                ):
                    output_grid[x][y] = surround_color
                elif (
                    output_grid[x][y] == 0
                    and y - 1 > 0
                    and output_grid[x][y - 1] == interior_color
                ):
                    output_grid[x][y] = surround_color

        input_grid = np.copy(output_grid)
        for x in range(grid_x_dim):
            for y in range(grid_y_dim):
                if input_grid[x][y] == interior_color:
                    input_grid[x][y] = 0

        if fill_exterior > 0:
            for x in range(grid_x_dim):
                for y in range(grid_y_dim):
                    if output_grid[x][y] == interior_color:
                        output_grid[x][y] = 0
                    elif output_grid[x][y] == 0:
                        output_grid[x][y] = interior_color

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

    json_string = json.dumps(task, cls=NumpyArrayEncoder)

    return json_string


def generate_gravity_puzzle(
    num_train_pairs, num_test_pairs, min_grid_dim, max_grid_dim
):
    input_grids = []
    output_grids = []

    rot_num = randint(0, 3)

    pair_count = 0
    w_loop_count = 0
    while pair_count < num_train_pairs + num_test_pairs:
        w_loop_count += 1
        if w_loop_count > 1000:
            return ""
        grid_x_dim = randint(min_grid_dim, max_grid_dim)
        grid_y_dim = randint(min_grid_dim, max_grid_dim)

        loop_again = False
        sub_grid_x_dim = randint(2, 7)
        sub_grid_y_dim = randint(2, 7)
        size = sub_grid_x_dim * sub_grid_y_dim
        permuted_array = np.random.randint(10, size=size)

        if check_zero_rows_cols(permuted_array, sub_grid_x_dim, sub_grid_y_dim):
            continue

        sub_grid = permuted_array.reshape((sub_grid_x_dim, sub_grid_y_dim))

        input_grid = np.zeros((grid_x_dim, grid_y_dim), dtype=int)
        # Place sub-grid randomly inside of input_grid
        rand_start_x_index = randint(0, grid_x_dim - sub_grid_x_dim)
        rand_start_y_index = randint(0, grid_y_dim - sub_grid_y_dim)
        input_grid[
            rand_start_x_index : rand_start_x_index + sub_grid_x_dim,
            rand_start_y_index : rand_start_y_index + sub_grid_y_dim,
        ] = sub_grid

        for x in range(grid_x_dim):
            for y in range(grid_y_dim):
                if (
                    x == 0 or y == 0 or x == grid_x_dim - 1 or y == grid_y_dim - 1
                ) and input_grid[x][y] != 0:
                    loop_again = True
                    break
                if loop_again:
                    break
        if loop_again:
            continue

        dist_to_ground = (grid_y_dim - 1) - (rand_start_y_index + sub_grid_y_dim)
        output_grid = np.zeros((grid_x_dim, grid_y_dim), dtype=int)
        output_grid[
            rand_start_x_index : rand_start_x_index + sub_grid_x_dim,
            0:sub_grid_y_dim,
        ] = sub_grid

        input_grid = np.rot90(input_grid, k=rot_num)
        input_grids.append(input_grid)

        output_grid = np.rot90(output_grid, k=rot_num)
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
        if i < 3:
            task["train"][i]["input"] = input_grid
            task["train"][i]["output"] = output_grid
        else:
            task["test"][0]["input"] = input_grid
            task["test"][0]["output"] = output_grid
        i += 1

    json_string = json.dumps(task, cls=NumpyArrayEncoder)

    return json_string


def generate_output_grid_size_equals_input_puzzle_easy(
    num_train_pairs, num_test_pairs, min_grid_dim, max_grid_dim
):
    input_grids = []
    output_grids = []

    fill_int = 0  # randint(0, 9)

    pair_count = 0
    w_loop_count = 0
    while pair_count < num_train_pairs + num_test_pairs:
        w_loop_count += 1
        if w_loop_count > 1000:
            return ""
        grid_x_dim = randint(min_grid_dim, max_grid_dim)
        grid_y_dim = randint(min_grid_dim, max_grid_dim)

        # output_grid = np.zeros((grid_x_dim, grid_y_dim), dtype=int)
        output_grid = np.full((grid_x_dim, grid_y_dim), fill_int, dtype=int)
        output_grids.append(output_grid)

        input_grid = np.copy(output_grid)
        for x in range(grid_x_dim):
            for y in range(grid_y_dim):
                input_grid[x][y] = randint(0, 9)
        input_grids.append(input_grid)

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

    # json_string = json.dumps(task)
    json_string = json.dumps(task, cls=NumpyArrayEncoder)

    return json_string


def generate_output_grid_size_equals_input_puzzle(
    num_train_pairs, num_test_pairs, min_grid_dim, max_grid_dim
):
    input_grids = []
    output_grids = []

    fill_int = randint(0, 9)

    pair_count = 0
    w_loop_count = 0
    while pair_count < num_train_pairs + num_test_pairs:
        w_loop_count += 1
        if w_loop_count > 1000:
            return ""
        grid_x_dim = randint(min_grid_dim, max_grid_dim)
        grid_y_dim = randint(min_grid_dim, max_grid_dim)

        # output_grid = np.zeros((grid_x_dim, grid_y_dim), dtype=int)
        output_grid = np.full((grid_x_dim, grid_y_dim), fill_int, dtype=int)
        output_grids.append(output_grid)

        input_grid = np.copy(output_grid)
        for x in range(grid_x_dim):
            for y in range(grid_y_dim):
                input_grid[x][y] = randint(0, 9)
        input_grids.append(input_grid)

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

    # json_string = json.dumps(task)
    json_string = json.dumps(task, cls=NumpyArrayEncoder)

    return json_string


def generate_isolate_obj_puzzle_easy(
    num_train_pairs, num_test_pairs, min_grid_dim, max_grid_dim
):
    sub_grid_x_dim = randint(2, 6)
    sub_grid_y_dim = randint(2, 6)
    size = sub_grid_x_dim * sub_grid_y_dim
    permuted_array = np.random.randint(10, size=size)

    while check_zero_rows_cols(permuted_array, sub_grid_x_dim, sub_grid_y_dim):
        sub_grid_x_dim = randint(2, 6)
        sub_grid_y_dim = randint(2, 6)
        size = sub_grid_x_dim * sub_grid_y_dim
        permuted_array = np.random.randint(10, size=size)

    sub_grid = permuted_array.reshape((sub_grid_x_dim, sub_grid_y_dim))

    input_grids = []
    output_grids = []
    flattened_isolated_obj_grids = set()

    pair_count = 0
    w_loop_count = 0
    while pair_count < num_train_pairs + num_test_pairs:
        w_loop_count += 1
        if w_loop_count > 1000:
            return ""
        grid_x_dim = randint(min_grid_dim, max_grid_dim)
        grid_y_dim = randint(min_grid_dim, max_grid_dim)

        isolated_obj_grid = np.zeros((grid_x_dim, grid_y_dim), dtype=int)
        # Place sub-grid randomly inside of isolated_obj_grid
        rand_start_x_index = randint(0, grid_x_dim - sub_grid_x_dim)
        rand_start_y_index = randint(0, grid_y_dim - sub_grid_y_dim)
        isolated_obj_grid[
            rand_start_x_index : rand_start_x_index + sub_grid_x_dim,
            rand_start_y_index : rand_start_y_index + sub_grid_y_dim,
        ] = sub_grid

        flat_tuple = tuple(isolated_obj_grid.flatten())

        # Add the matrix to the set if it's not already present
        if flat_tuple not in flattened_isolated_obj_grids:
            pair_count += 1

            flattened_isolated_obj_grids.add(flat_tuple)

            output_grid = np.copy(isolated_obj_grid)
            output_grids.append(output_grid)

            make_input_grid = True
            w2_loop_count = 0
            while make_input_grid:
                w2_loop_count += 1
                if w2_loop_count > 1000:
                    return ""
                make_input_grid = False
                # print("input_grid", pair_count)
                input_grid = np.copy(isolated_obj_grid)
                random_bg_color = randint(1, 9)
                for x in range(grid_x_dim):
                    for y in range(grid_y_dim):
                        if not (
                            rand_start_x_index
                            <= x
                            < rand_start_x_index + sub_grid_x_dim
                            and rand_start_y_index
                            <= y
                            < rand_start_y_index + sub_grid_y_dim
                        ):
                            input_grid[x][y] = random_bg_color  # randint(0, 9)

                main_shape = input_grid.shape
                sub_shape = sub_grid.shape
                # Calculate the possible positions to look for the sub_array in main_array
                position_range = np.subtract(main_shape, sub_shape) + 1
                # Loop over all possible positions in main_array
                num_occurances_of_sub_array = 0

                for i in range(position_range[0]):
                    for j in range(position_range[1]):
                        # Extract the candidate sub_array from main_array
                        duplicate_sub_grid_candidate = input_grid[
                            i : i + sub_shape[0], j : j + sub_shape[1]
                        ]
                        # Compare the candidate sub_array with the sub_array
                        if np.array_equal(duplicate_sub_grid_candidate, sub_grid):
                            num_occurances_of_sub_array += 1

                if num_occurances_of_sub_array > 1:
                    # print("num_occurances_of_sub_array > 1")
                    make_input_grid = True
                else:
                    input_grids.append(input_grid)

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

    json_string = json.dumps(task, cls=NumpyArrayEncoder)

    return json_string


def generate_isolate_obj_puzzle_med(
    num_train_pairs, num_test_pairs, min_grid_dim, max_grid_dim
):
    sub_grid_x_dim = randint(2, 6)
    sub_grid_y_dim = randint(2, 6)
    size = sub_grid_x_dim * sub_grid_y_dim
    permuted_array = np.random.randint(10, size=size)

    while check_zero_rows_cols(permuted_array, sub_grid_x_dim, sub_grid_y_dim):
        sub_grid_x_dim = randint(2, 6)
        sub_grid_y_dim = randint(2, 6)
        size = sub_grid_x_dim * sub_grid_y_dim
        permuted_array = np.random.randint(10, size=size)

    sub_grid = permuted_array.reshape((sub_grid_x_dim, sub_grid_y_dim))

    input_grids = []
    output_grids = []
    flattened_isolated_obj_grids = set()

    pair_count = 0
    w_loop_count = 0
    while pair_count < num_train_pairs + num_test_pairs:
        w_loop_count += 1
        if w_loop_count > 1000:
            return ""
        grid_x_dim = randint(min_grid_dim, max_grid_dim)
        grid_y_dim = randint(min_grid_dim, max_grid_dim)

        isolated_obj_grid = np.zeros((grid_x_dim, grid_y_dim), dtype=int)
        # Place sub-grid randomly inside of isolated_obj_grid
        rand_start_x_index = randint(0, grid_x_dim - sub_grid_x_dim)
        rand_start_y_index = randint(0, grid_y_dim - sub_grid_y_dim)
        isolated_obj_grid[
            rand_start_x_index : rand_start_x_index + sub_grid_x_dim,
            rand_start_y_index : rand_start_y_index + sub_grid_y_dim,
        ] = sub_grid

        flat_tuple = tuple(isolated_obj_grid.flatten())

        # Add the matrix to the set if it's not already present
        if flat_tuple not in flattened_isolated_obj_grids:
            pair_count += 1

            flattened_isolated_obj_grids.add(flat_tuple)

            output_grid = np.copy(isolated_obj_grid)
            # for x in range(grid_x_dim):
            #     for y in range(grid_y_dim):
            #         if output_grid[x][y] == 0:
            #             output_grid[x][y] = output_bg_color
            output_grids.append(output_grid)

            make_input_grid = True
            w2_loop_count = 0
            while make_input_grid:
                w2_loop_count += 1
                if w2_loop_count > 1000:
                    return ""
                make_input_grid = False
                # print("input_grid", pair_count)
                input_grid = np.copy(isolated_obj_grid)
                random_bg_color1 = randint(1, 9)
                random_bg_color2 = randint(1, 9)
                while random_bg_color2 == random_bg_color1:
                    random_bg_color2 = randint(1, 9)
                for x in range(grid_x_dim):
                    for y in range(grid_y_dim):
                        if not (
                            rand_start_x_index
                            <= x
                            < rand_start_x_index + sub_grid_x_dim
                            and rand_start_y_index
                            <= y
                            < rand_start_y_index + sub_grid_y_dim
                        ):
                            if random_bg_color1 < random_bg_color2:
                                input_grid[x][y] = randint(
                                    random_bg_color1, random_bg_color2
                                )
                            else:
                                input_grid[x][y] = randint(
                                    random_bg_color2, random_bg_color1
                                )

                main_shape = input_grid.shape
                sub_shape = sub_grid.shape
                # Calculate the possible positions to look for the sub_array in main_array
                position_range = np.subtract(main_shape, sub_shape) + 1
                # Loop over all possible positions in main_array
                num_occurances_of_sub_array = 0

                for i in range(position_range[0]):
                    for j in range(position_range[1]):
                        # Extract the candidate sub_array from main_array
                        duplicate_sub_grid_candidate = input_grid[
                            i : i + sub_shape[0], j : j + sub_shape[1]
                        ]
                        # Compare the candidate sub_array with the sub_array
                        if np.array_equal(duplicate_sub_grid_candidate, sub_grid):
                            num_occurances_of_sub_array += 1

                if num_occurances_of_sub_array > 1:
                    # print("num_occurances_of_sub_array > 1")
                    make_input_grid = True
                else:
                    input_grids.append(input_grid)

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

    json_string = json.dumps(task, cls=NumpyArrayEncoder)

    return json_string


def generate_isolate_obj_puzzle(
    num_train_pairs, num_test_pairs, min_grid_dim, max_grid_dim
):
    sub_grid_x_dim = randint(2, 6)
    sub_grid_y_dim = randint(2, 6)
    size = sub_grid_x_dim * sub_grid_y_dim
    permuted_array = np.random.randint(10, size=size)

    while check_zero_rows_cols(permuted_array, sub_grid_x_dim, sub_grid_y_dim):
        sub_grid_x_dim = randint(2, 6)
        sub_grid_y_dim = randint(2, 6)
        size = sub_grid_x_dim * sub_grid_y_dim
        permuted_array = np.random.randint(10, size=size)

    sub_grid = permuted_array.reshape((sub_grid_x_dim, sub_grid_y_dim))

    input_grids = []
    output_grids = []
    flattened_isolated_obj_grids = set()

    pair_count = 0
    w_loop_count = 0
    while pair_count < num_train_pairs + num_test_pairs:
        w_loop_count += 1
        if w_loop_count > 1000:
            return ""
        grid_x_dim = randint(min_grid_dim, max_grid_dim)
        grid_y_dim = randint(min_grid_dim, max_grid_dim)

        isolated_obj_grid = np.zeros((grid_x_dim, grid_y_dim), dtype=int)
        # Place sub-grid randomly inside of isolated_obj_grid
        rand_start_x_index = randint(0, grid_x_dim - sub_grid_x_dim)
        rand_start_y_index = randint(0, grid_y_dim - sub_grid_y_dim)
        isolated_obj_grid[
            rand_start_x_index : rand_start_x_index + sub_grid_x_dim,
            rand_start_y_index : rand_start_y_index + sub_grid_y_dim,
        ] = sub_grid

        flat_tuple = tuple(isolated_obj_grid.flatten())

        # Add the matrix to the set if it's not already present
        if flat_tuple not in flattened_isolated_obj_grids:
            pair_count += 1

            flattened_isolated_obj_grids.add(flat_tuple)

            output_grid = np.copy(isolated_obj_grid)
            # for x in range(grid_x_dim):
            #     for y in range(grid_y_dim):
            #         if output_grid[x][y] == 0:
            #             output_grid[x][y] = output_bg_color
            output_grids.append(output_grid)

            make_input_grid = True
            w2_loop_count = 0
            while make_input_grid:
                w2_loop_count += 1
                if w2_loop_count > 1000:
                    return ""
                make_input_grid = False
                # print("input_grid", pair_count)
                input_grid = np.copy(isolated_obj_grid)
                for x in range(grid_x_dim):
                    for y in range(grid_y_dim):
                        if not (
                            rand_start_x_index
                            <= x
                            < rand_start_x_index + sub_grid_x_dim
                            and rand_start_y_index
                            <= y
                            < rand_start_y_index + sub_grid_y_dim
                        ):
                            input_grid[x][y] = randint(0, 9)

                main_shape = input_grid.shape
                sub_shape = sub_grid.shape
                # Calculate the possible positions to look for the sub_array in main_array
                position_range = np.subtract(main_shape, sub_shape) + 1
                # Loop over all possible positions in main_array
                num_occurances_of_sub_array = 0

                for i in range(position_range[0]):
                    for j in range(position_range[1]):
                        # Extract the candidate sub_array from main_array
                        duplicate_sub_grid_candidate = input_grid[
                            i : i + sub_shape[0], j : j + sub_shape[1]
                        ]
                        # Compare the candidate sub_array with the sub_array
                        if np.array_equal(duplicate_sub_grid_candidate, sub_grid):
                            num_occurances_of_sub_array += 1

                if num_occurances_of_sub_array > 1:
                    # print("num_occurances_of_sub_array > 1")
                    make_input_grid = True
                else:
                    input_grids.append(input_grid)

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

    json_string = json.dumps(task, cls=NumpyArrayEncoder)

    return json_string


def generate_largest_smallest_obj_puzzle_v1(
    num_train_pairs, num_test_pairs, min_grid_dim, max_grid_dim
):
    input_grids = []
    output_grids = []

    keep_larger = randint(0, 1)

    rand_rot_or_flip = randint(0, 5)

    pair_count = 0
    w_loop_count = 0
    while pair_count < num_train_pairs + num_test_pairs:
        w_loop_count += 1
        if w_loop_count > 1000:
            return ""
        grid_x_dim = randint(min_grid_dim, max_grid_dim)
        grid_y_dim = randint(min_grid_dim, max_grid_dim)
        #################### obj 1 ##################################################
        sub_grid_1_x_dim = randint(2, 4)
        sub_grid_1_y_dim = randint(2, 4)
        permuted_array_1 = np.random.randint(
            10, size=sub_grid_1_x_dim * sub_grid_1_y_dim
        )

        if check_zero_rows_cols(permuted_array_1, sub_grid_1_x_dim, sub_grid_1_y_dim):
            continue

        sub_grid_1 = permuted_array_1.reshape((sub_grid_1_x_dim, sub_grid_1_y_dim))

        sub_grid_1_square_count = 0
        for x in range(sub_grid_1_x_dim):
            for y in range(sub_grid_1_y_dim):
                if sub_grid_1[x][y] != 0:
                    sub_grid_1_square_count += 1

        isolated_obj_grid_1 = np.zeros((grid_x_dim, grid_y_dim), dtype=int)

        start_x_index = randint(0, ((grid_x_dim - (sub_grid_1_x_dim * 2)) // 2))
        start_y_index = randint(0, ((grid_y_dim - (sub_grid_1_y_dim * 2)) // 2))

        isolated_obj_grid_1[
            start_x_index : start_x_index + sub_grid_1_x_dim,
            start_y_index : start_y_index + sub_grid_1_y_dim,
        ] = sub_grid_1

        #################### obj 2 ##################################################
        black_row_or_column = True
        same_square_count = True
        overlapping = True
        w2_loop_count = 0
        while black_row_or_column or same_square_count or overlapping:
            w2_loop_count += 1
            if w2_loop_count > 1000:
                return ""
            black_row_or_column = False
            same_square_count = False
            overlapping = False

            sub_grid_2_x_dim = randint(2, 4)
            sub_grid_2_y_dim = randint(2, 4)

            permuted_array_2 = np.random.randint(
                10, size=sub_grid_2_x_dim * sub_grid_2_y_dim
            )

            if check_zero_rows_cols(
                permuted_array_2, sub_grid_2_x_dim, sub_grid_2_y_dim
            ):
                black_row_or_column = True
                continue

            sub_grid_2 = permuted_array_2.reshape((sub_grid_2_x_dim, sub_grid_2_y_dim))

            sub_grid_2_square_count = 0
            for x in range(sub_grid_2_x_dim):
                for y in range(sub_grid_2_y_dim):
                    if sub_grid_2[x][y] != 0:
                        sub_grid_2_square_count += 1

            if sub_grid_2_square_count == sub_grid_1_square_count:
                same_square_count = True
                continue

            isolated_obj_grid_2 = np.zeros((grid_x_dim, grid_y_dim), dtype=int)

            start_x_index_2 = randint(0, ((grid_x_dim - (sub_grid_2_x_dim * 2)) // 2))
            start_y_index_2 = randint(0, ((grid_y_dim - (sub_grid_2_y_dim * 2)) // 2))

            isolated_obj_grid_2[
                start_x_index_2 : start_x_index_2 + sub_grid_2_x_dim,
                start_y_index_2 : start_y_index_2 + sub_grid_2_y_dim,
            ] = sub_grid_2

            isolated_obj_grid_2 = np.flipud(isolated_obj_grid_2)
            isolated_obj_grid_2 = np.fliplr(isolated_obj_grid_2)

            break_all = False
            for x in range(grid_x_dim):
                for y in range(grid_y_dim):
                    if (
                        isolated_obj_grid_1[x][y] != 0
                        and isolated_obj_grid_2[x][y] != 0
                    ):
                        overlapping = True
                        break_all = True
                        break
                    elif (
                        x + 1 < grid_x_dim - 1
                        and isolated_obj_grid_1[x][y] != 0
                        and isolated_obj_grid_2[x + 1][y] != 0
                    ):
                        overlapping = True
                        break_all = True
                        break
                    elif (
                        y + 1 < grid_y_dim - 1
                        and isolated_obj_grid_1[x][y] != 0
                        and isolated_obj_grid_2[x][y + 1] != 0
                    ):
                        overlapping = True
                        break_all = True
                        break
                    elif (
                        x + 1 < grid_y_dim - 1
                        and y + 1 < grid_y_dim - 1
                        and isolated_obj_grid_1[x][y] != 0
                        and isolated_obj_grid_2[x + 1][y + 1] != 0
                    ):
                        overlapping = True
                        break_all = True
                        break
                if break_all:
                    break
            if overlapping:
                continue

        input_grid = np.copy(isolated_obj_grid_1)
        for x in range(grid_x_dim):
            for y in range(grid_y_dim):
                if isolated_obj_grid_2[x][y] != 0:
                    input_grid[x][y] = isolated_obj_grid_2[x][y]

        if keep_larger > 0:
            if sub_grid_1_square_count > sub_grid_2_square_count:
                output_grid = np.copy(sub_grid_1)
            else:
                output_grid = np.copy(sub_grid_2)
        else:
            if sub_grid_1_square_count < sub_grid_2_square_count:
                output_grid = np.copy(sub_grid_1)
            else:
                output_grid = np.copy(sub_grid_2)

        #####################################################################################
        if rand_rot_or_flip > 0 and rand_rot_or_flip < 4:
            input_grid = np.rot90(input_grid, k=rand_rot_or_flip)
            output_grid = np.rot90(output_grid, k=rand_rot_or_flip)
        elif rand_rot_or_flip == 4:
            input_grid = np.fliplr(input_grid)
            output_grid = np.fliplr(output_grid)
        elif rand_rot_or_flip == 5:
            input_grid = np.flipud(input_grid)
            output_grid = np.flipud(output_grid)

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

    json_string = json.dumps(task, cls=NumpyArrayEncoder)

    return json_string


def generate_largest_smallest_obj_puzzle_v2(
    num_train_pairs, num_test_pairs, min_grid_dim, max_grid_dim
):
    input_grids = []
    output_grids = []

    keep_larger = randint(0, 1)

    rand_rot_or_flip = randint(0, 5)
    rand_scale_or_shrink = randint(0, 100)

    pair_count = 0
    w_loop_count = 0
    while pair_count < num_train_pairs + num_test_pairs:
        w_loop_count += 1
        if w_loop_count > 1000:
            return ""
        grid_x_dim = randint(min_grid_dim, max_grid_dim)
        grid_y_dim = randint(min_grid_dim, max_grid_dim)
        #################### obj 1 ##################################################
        sub_grid_1_x_dim = randint(2, 4)
        sub_grid_1_y_dim = randint(2, 4)
        permuted_array_1 = np.random.randint(
            10, size=sub_grid_1_x_dim * sub_grid_1_y_dim
        )

        if check_zero_rows_cols(permuted_array_1, sub_grid_1_x_dim, sub_grid_1_y_dim):
            continue

        sub_grid_1 = permuted_array_1.reshape((sub_grid_1_x_dim, sub_grid_1_y_dim))

        sub_grid_1_square_count = 0
        for x in range(sub_grid_1_x_dim):
            for y in range(sub_grid_1_y_dim):
                if sub_grid_1[x][y] != 0:
                    sub_grid_1_square_count += 1

        isolated_obj_grid_1 = np.zeros((grid_x_dim, grid_y_dim), dtype=int)

        start_x_index = randint(0, ((grid_x_dim - (sub_grid_1_x_dim * 2)) // 2))
        start_y_index = randint(0, ((grid_y_dim - (sub_grid_1_y_dim * 2)) // 2))

        isolated_obj_grid_1[
            start_x_index : start_x_index + sub_grid_1_x_dim,
            start_y_index : start_y_index + sub_grid_1_y_dim,
        ] = sub_grid_1

        #################### obj 2 ##################################################
        black_row_or_column = True
        same_square_count = True
        overlapping = True
        w2_loop_count = 0
        while black_row_or_column or same_square_count or overlapping:
            w2_loop_count += 1
            if w2_loop_count > 1000:
                return ""
            black_row_or_column = False
            same_square_count = False
            overlapping = False

            sub_grid_2_x_dim = randint(2, 4)
            sub_grid_2_y_dim = randint(2, 4)

            permuted_array_2 = np.random.randint(
                10, size=sub_grid_2_x_dim * sub_grid_2_y_dim
            )

            if check_zero_rows_cols(
                permuted_array_2, sub_grid_2_x_dim, sub_grid_2_y_dim
            ):
                black_row_or_column = True
                continue

            sub_grid_2 = permuted_array_2.reshape((sub_grid_2_x_dim, sub_grid_2_y_dim))

            sub_grid_2_square_count = 0
            for x in range(sub_grid_2_x_dim):
                for y in range(sub_grid_2_y_dim):
                    if sub_grid_2[x][y] != 0:
                        sub_grid_2_square_count += 1

            if sub_grid_2_square_count == sub_grid_1_square_count:
                same_square_count = True
                continue

            isolated_obj_grid_2 = np.zeros((grid_x_dim, grid_y_dim), dtype=int)

            start_x_index_2 = randint(0, ((grid_x_dim - (sub_grid_2_x_dim * 2)) // 2))
            start_y_index_2 = randint(0, ((grid_y_dim - (sub_grid_2_y_dim * 2)) // 2))

            isolated_obj_grid_2[
                start_x_index_2 : start_x_index_2 + sub_grid_2_x_dim,
                start_y_index_2 : start_y_index_2 + sub_grid_2_y_dim,
            ] = sub_grid_2

            isolated_obj_grid_2 = np.flipud(isolated_obj_grid_2)
            isolated_obj_grid_2 = np.fliplr(isolated_obj_grid_2)

            break_all = False
            for x in range(grid_x_dim):
                for y in range(grid_y_dim):
                    if (
                        isolated_obj_grid_1[x][y] != 0
                        and isolated_obj_grid_2[x][y] != 0
                    ):
                        overlapping = True
                        break_all = True
                        break
                    elif (
                        x + 1 < grid_x_dim - 1
                        and isolated_obj_grid_1[x][y] != 0
                        and isolated_obj_grid_2[x + 1][y] != 0
                    ):
                        overlapping = True
                        break_all = True
                        break
                    elif (
                        y + 1 < grid_y_dim - 1
                        and isolated_obj_grid_1[x][y] != 0
                        and isolated_obj_grid_2[x][y + 1] != 0
                    ):
                        overlapping = True
                        break_all = True
                        break
                    elif (
                        x + 1 < grid_y_dim - 1
                        and y + 1 < grid_y_dim - 1
                        and isolated_obj_grid_1[x][y] != 0
                        and isolated_obj_grid_2[x + 1][y + 1] != 0
                    ):
                        overlapping = True
                        break_all = True
                        break
                if break_all:
                    break
            if overlapping:
                continue

        input_grid = np.copy(isolated_obj_grid_1)
        for x in range(grid_x_dim):
            for y in range(grid_y_dim):
                if isolated_obj_grid_2[x][y] != 0:
                    input_grid[x][y] = isolated_obj_grid_2[x][y]

        if keep_larger > 0:
            if sub_grid_1_square_count > sub_grid_2_square_count:
                output_grid = np.copy(isolated_obj_grid_1)
            else:
                output_grid = np.copy(isolated_obj_grid_2)
        else:
            if sub_grid_1_square_count < sub_grid_2_square_count:
                output_grid = np.copy(isolated_obj_grid_1)
            else:
                output_grid = np.copy(isolated_obj_grid_2)

        #####################################################################################
        if rand_rot_or_flip > 0 and rand_rot_or_flip < 4:
            input_grid = np.rot90(input_grid, k=rand_rot_or_flip)
            output_grid = np.rot90(output_grid, k=rand_rot_or_flip)
        elif rand_rot_or_flip == 4:
            input_grid = np.fliplr(input_grid)
            output_grid = np.fliplr(output_grid)
        elif rand_rot_or_flip == 5:
            input_grid = np.flipud(input_grid)
            output_grid = np.flipud(output_grid)

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

    json_string = json.dumps(task, cls=NumpyArrayEncoder)

    return json_string


def generate_mask_puzzle(num_train_pairs, num_test_pairs, min_grid_dim, max_grid_dim):
    sub_grid_x_dim = randint(2, 6)
    sub_grid_y_dim = randint(2, 6)
    size = sub_grid_x_dim * sub_grid_y_dim
    permuted_array = np.random.randint(10, size=size)

    while check_zero_rows_cols(permuted_array, sub_grid_x_dim, sub_grid_y_dim):
        sub_grid_x_dim = randint(2, 6)
        sub_grid_y_dim = randint(2, 6)
        size = sub_grid_x_dim * sub_grid_y_dim
        permuted_array = np.random.randint(10, size=size)

    sub_grid = permuted_array.reshape((sub_grid_x_dim, sub_grid_y_dim))

    input_grids = []
    isolated_obj_grids = []
    output_grids = []
    flattened_isolated_obj_grids = set()
    # mask_color = randint(0, 9)
    mask_color = 0

    pair_count = 0
    w_loop_count = 0
    while pair_count < num_train_pairs + num_test_pairs:
        w_loop_count += 1
        if w_loop_count > 1000:
            return ""
        grid_x_dim = randint(min_grid_dim, max_grid_dim)
        grid_y_dim = randint(min_grid_dim, max_grid_dim)

        isolated_obj_grid = np.zeros((grid_x_dim, grid_y_dim), dtype=int)
        # Place sub-grid randomly inside of output_grid
        rand_start_x_index = randint(0, grid_x_dim - sub_grid_x_dim)
        rand_start_y_index = randint(0, grid_y_dim - sub_grid_y_dim)
        isolated_obj_grid[
            rand_start_x_index : rand_start_x_index + sub_grid_x_dim,
            rand_start_y_index : rand_start_y_index + sub_grid_y_dim,
        ] = sub_grid

        flat_tuple = tuple(isolated_obj_grid.flatten())

        # Add the matrix to the set if it's not already present
        if flat_tuple not in flattened_isolated_obj_grids:
            flattened_isolated_obj_grids.add(flat_tuple)
            isolated_obj_grids.append(isolated_obj_grid)
            pair_count += 1

            make_input_grid = True
            w2_loop_count = 0
            while make_input_grid:
                w2_loop_count += 1
                if w2_loop_count > 1000:
                    return ""
                make_input_grid = False
                # print("input_grid", pair_count)
                input_grid = np.copy(isolated_obj_grid)
                for x in range(grid_x_dim):
                    for y in range(grid_y_dim):
                        if not (
                            rand_start_x_index
                            <= x
                            < rand_start_x_index + sub_grid_x_dim
                            and rand_start_y_index
                            <= y
                            < rand_start_y_index + sub_grid_y_dim
                        ):
                            input_grid[x][y] = randint(0, 9)

                main_shape = input_grid.shape
                sub_shape = sub_grid.shape
                # Calculate the possible positions to look for the sub_array in main_array
                position_range = np.subtract(main_shape, sub_shape) + 1
                # Loop over all possible positions in main_array
                num_occurances_of_sub_array = 0

                for i in range(position_range[0]):
                    for j in range(position_range[1]):
                        # Extract the candidate sub_array from main_array
                        duplicate_sub_grid_candidate = input_grid[
                            i : i + sub_shape[0], j : j + sub_shape[1]
                        ]
                        # Compare the candidate sub_array with the sub_array
                        if np.array_equal(duplicate_sub_grid_candidate, sub_grid):
                            num_occurances_of_sub_array += 1

                if num_occurances_of_sub_array > 1:
                    # print("num_occurances_of_sub_array > 1")
                    make_input_grid = True
                else:
                    input_grids.append(input_grid)
                    output_grid = np.copy(input_grid)
                    for x in range(grid_x_dim):
                        for y in range(grid_y_dim):
                            if isolated_obj_grid[x][y] > 0:
                                output_grid[x][y] = mask_color
                    output_grids.append(output_grid)

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

    json_string = json.dumps(task, cls=NumpyArrayEncoder)

    return json_string


def generate_mirror_grid_puzzle(
    num_train_pairs, num_test_pairs, min_grid_dim, max_grid_dim
):
    input_grids = []
    output_grids = []
    flip_type = randint(0, 1)

    pair_count = 0
    w_loop_count = 0
    while pair_count < num_train_pairs + num_test_pairs:
        w_loop_count += 1
        if w_loop_count > 1000:
            return ""
        grid_x_dim = randint(min_grid_dim, max_grid_dim)
        grid_y_dim = randint(min_grid_dim, max_grid_dim)

        pair_count += 1
        input_grid = np.zeros((grid_x_dim, grid_y_dim), dtype=int)
        for x in range(grid_x_dim):
            for y in range(grid_y_dim):
                if input_grid[x][y] == 0:
                    input_grid[x][y] = randint(0, 9)
        input_grids.append(input_grid)

        output_grid = np.copy(input_grid)
        # output_grid = np.rot90(output_grid, k=rot_num)
        if flip_type == 0:
            output_grid = np.flipud(output_grid)
        else:
            output_grid = np.fliplr(output_grid)
        output_grids.append(output_grid)

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

    json_string = json.dumps(task, cls=NumpyArrayEncoder)

    return json_string


def generate_mirror_obj_puzzle(
    num_train_pairs, num_test_pairs, min_grid_dim, max_grid_dim
):
    sub_grid_x_dim = randint(2, 6)
    sub_grid_y_dim = randint(2, 6)
    size = sub_grid_x_dim * sub_grid_y_dim
    permuted_array = np.random.randint(10, size=size)

    while check_zero_rows_cols(permuted_array, sub_grid_x_dim, sub_grid_y_dim):
        sub_grid_x_dim = randint(2, 6)
        sub_grid_y_dim = randint(2, 6)
        size = sub_grid_x_dim * sub_grid_y_dim
        permuted_array = np.random.randint(10, size=size)

    sub_grid = permuted_array.reshape((sub_grid_x_dim, sub_grid_y_dim))

    input_grids = []
    base_grids = []
    output_grids = []
    flattened_base_grids = set()

    flip_type = randint(0, 1)
    if flip_type == 0:
        sub_grid_mirrored = np.flipud(sub_grid)
    else:
        sub_grid_mirrored = np.fliplr(sub_grid)

    pair_count = 0
    w_loop_count = 0
    while pair_count < num_train_pairs + num_test_pairs:
        w_loop_count += 1
        if w_loop_count > 1000:
            return ""
        grid_x_dim = randint(min_grid_dim, max_grid_dim)
        grid_y_dim = randint(min_grid_dim, max_grid_dim)

        base_grid = np.random.randint(10, size=(grid_x_dim, grid_y_dim))
        base_grid_mirrored = np.copy(base_grid)

        # Place sub-grid randomly inside of base_grid
        rand_start_x_index = randint(0, grid_x_dim - sub_grid_x_dim)
        rand_start_y_index = randint(0, grid_y_dim - sub_grid_y_dim)
        base_grid[
            rand_start_x_index : rand_start_x_index + sub_grid_x_dim,
            rand_start_y_index : rand_start_y_index + sub_grid_y_dim,
        ] = sub_grid

        flat_tuple = tuple(base_grid.flatten())

        # Add the matrix to the set if it's not already present
        if flat_tuple not in flattened_base_grids:
            flattened_base_grids.add(flat_tuple)

            base_grid_mirrored[
                rand_start_x_index : rand_start_x_index + sub_grid_x_dim,
                rand_start_y_index : rand_start_y_index + sub_grid_y_dim,
            ] = sub_grid_mirrored

            input_grid = np.copy(base_grid)
            input_grids.append(input_grid)

            output_grid = np.copy(base_grid_mirrored)
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

    json_string = json.dumps(task, cls=NumpyArrayEncoder)

    return json_string


def most_frequent_color_in_row(find_min_color, rand_rot, min_grid_dim, max_grid_dim):
    find_min_color = 0
    do_again = True
    do_again_count = 0
    while do_again:
        do_again_count += 1
        if do_again_count > 1000:
            return [], []
        do_again = False
        # Define a grid with a random distribution of colors
        grid_x_dim = randint(min_grid_dim, max_grid_dim)
        grid_y_dim = randint(min_grid_dim, max_grid_dim)
        input_grid = np.random.choice(
            range(0, 10), size=(grid_x_dim, grid_y_dim)
        ).tolist()

        # Generate the output grid where each color i becomes the most frequently
        # occurring color in the i-th row of the input grid
        output_grid = []
        for row in input_grid:
            possible_colors_in_row = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
            color_counts = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            for i, color in enumerate(possible_colors_in_row):
                for square in row:
                    if square == color:
                        color_counts[i] += 1

            output_row_color = 0

            if find_min_color > 0:
                min_color_count = 9
                min_color = 9
                for i, color_count in enumerate(color_counts):
                    if color_count < min_color_count:
                        min_color = possible_colors_in_row[i]
                        min_color_count = color_count

                dup_min_colors = 0
                for color_count in color_counts:
                    if color_count == min_color_count:
                        dup_min_colors += 1
                        if dup_min_colors > 1:
                            do_again = True
                            break
                output_row_color = min_color
            else:
                max_color_count = 0
                max_color = 0
                for i, color_count in enumerate(color_counts):
                    if color_count > max_color_count:
                        max_color = possible_colors_in_row[i]
                        max_color_count = color_count

                dup_max_colors = 0
                for color_count in color_counts:
                    if color_count == max_color_count:
                        dup_max_colors += 1
                        if dup_max_colors > 1:
                            do_again = True
                            break

                output_row_color = max_color

            if do_again:
                break

            output_grid.append([output_row_color])

    #####################################################################################
    if rand_rot > 0:
        input_grid = np.rot90(input_grid, k=1)
        output_grid = np.rot90(output_grid, k=1)

    return input_grid, output_grid


def generate_most_freq_in_row_puzzle(
    num_train_pairs, num_test_pairs, min_grid_dim, max_grid_dim
):
    input_grids = []
    output_grids = []

    rand_rot = randint(0, 1)

    find_min_color = 0  # randint(0, 1)

    pair_count = 0
    w_loop_count = 0
    while pair_count < num_train_pairs + num_test_pairs:
        w_loop_count += 1
        if w_loop_count > 1000:
            return ""
        input_grid, output_grid = most_frequent_color_in_row(
            find_min_color, rand_rot, min_grid_dim, max_grid_dim
        )
        if len(input_grid) < 1 or len(output_grid) < 1:
            continue

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
        if i < 3:
            task["train"][i]["input"] = input_grid
            task["train"][i]["output"] = output_grid
        else:
            task["test"][0]["input"] = input_grid
            task["test"][0]["output"] = output_grid
        i += 1

    json_string = json.dumps(task, cls=NumpyArrayEncoder)

    return json_string


def generate_move_obj_puzzle(
    num_train_pairs, num_test_pairs, min_grid_dim, max_grid_dim
):
    input_grids = []
    output_grids = []

    random_move_x = randint(-2, 2)
    random_move_y = randint(-2, 2)
    if random_move_x == 0:
        while random_move_y == 0:
            random_move_y = randint(-2, 2)

    pair_count = 0
    redo = False
    w_loop_count = 0
    while pair_count < num_train_pairs + num_test_pairs or redo:
        w_loop_count += 1
        if w_loop_count > 1000:
            return ""
        grid_x_dim = randint(min_grid_dim, max_grid_dim)
        while grid_x_dim % 2 > 0:
            grid_x_dim = randint(min_grid_dim, max_grid_dim)
        grid_y_dim = randint(min_grid_dim, max_grid_dim)
        while grid_y_dim % 2 > 0:
            grid_y_dim = randint(min_grid_dim, max_grid_dim)

        redo = False

        sub_grid_x_dim = randint(2, 6)
        sub_grid_y_dim = randint(2, 6)
        size = sub_grid_x_dim * sub_grid_y_dim
        permuted_array = np.random.randint(10, size=size)

        while check_zero_rows_cols(permuted_array, sub_grid_x_dim, sub_grid_y_dim):
            sub_grid_x_dim = randint(2, 6)
            sub_grid_y_dim = randint(2, 6)
            size = sub_grid_x_dim * sub_grid_y_dim
            permuted_array = np.random.randint(10, size=size)

        sub_grid = permuted_array.reshape((sub_grid_x_dim, sub_grid_y_dim))

        isolated_obj_grid = np.zeros((grid_x_dim, grid_y_dim), dtype=int)
        # Place sub-grid randomly inside of isolated_obj_grid
        sub_grid = permuted_array.reshape((sub_grid_x_dim, sub_grid_y_dim))
        rand_start_x_index = randint(0, grid_x_dim - sub_grid_x_dim)
        rand_start_y_index = randint(0, grid_y_dim - sub_grid_y_dim)
        isolated_obj_grid[
            rand_start_x_index : rand_start_x_index + sub_grid_x_dim,
            rand_start_y_index : rand_start_y_index + sub_grid_y_dim,
        ] = sub_grid

        start_x = rand_start_x_index + random_move_x
        start_y = rand_start_y_index + random_move_y

        if start_x < 0 or start_y < 0:
            redo = True
        else:
            input_grid = np.copy(isolated_obj_grid)
            input_grids.append(input_grid)

            moved_obj_grid = np.zeros((2 * grid_x_dim, 2 * grid_y_dim), dtype=int)
            moved_obj_grid[
                start_x : start_x + sub_grid_x_dim,
                start_y : start_y + sub_grid_y_dim,
            ] = sub_grid

            moved_obj_grid = moved_obj_grid[:grid_x_dim, :grid_y_dim]

            output_grid = np.copy(moved_obj_grid)
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

    # json_string = json.dumps(task)
    json_string = json.dumps(task, cls=NumpyArrayEncoder)

    return json_string


def generate_move_obj_puzzle_easy(
    num_train_pairs, num_test_pairs, min_grid_dim, max_grid_dim
):
    input_grids = []
    output_grids = []

    random_move = randint(-1, 1)
    if random_move == 0:
        random_move = randint(-1, 1)

    x_or_y_move = randint(0, 1)

    pair_count = 0
    redo = False
    w_loop_count = 0
    while pair_count < num_train_pairs + num_test_pairs or redo:
        w_loop_count += 1
        if w_loop_count > 1000:
            return ""
        redo = False

        square_color = randint(1, 9)

        grid_x_dim = randint(min_grid_dim, max_grid_dim)
        grid_y_dim = randint(min_grid_dim, max_grid_dim)

        sub_grid_x_dim = 1
        sub_grid_y_dim = 1

        size = 1
        permuted_array = np.array([square_color])

        input_grid = np.zeros((grid_x_dim, grid_y_dim), dtype=int)
        rand_start_x_index = randint(0, grid_x_dim - 1)
        rand_start_y_index = randint(0, grid_y_dim - 1)

        input_grid[rand_start_x_index, rand_start_y_index] = square_color

        if x_or_y_move > 0:
            moved_x_index = rand_start_x_index + random_move
            moved_y_index = rand_start_y_index
        else:
            moved_x_index = rand_start_x_index
            moved_y_index = rand_start_y_index + random_move

        if (
            moved_x_index < 0
            or moved_y_index < 0
            or moved_x_index > grid_x_dim - 1
            or moved_y_index > grid_y_dim - 1
        ):
            redo = True
        else:
            input_grids.append(input_grid)

            output_grid = np.zeros((grid_x_dim, grid_y_dim), dtype=int)
            output_grid[moved_x_index, moved_y_index] = square_color
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

    # json_string = json.dumps(task)
    json_string = json.dumps(task, cls=NumpyArrayEncoder)

    return json_string


def generate_move_obj_puzzle_super_easy(
    num_train_pairs, num_test_pairs, min_grid_dim, max_grid_dim
):
    input_grids = []
    output_grids = []

    random_move = randint(-1, 1)
    if random_move == 0:
        random_move = randint(-1, 1)

    x_or_y_move = randint(0, 1)

    square_color = randint(1, 9)

    pair_count = 0
    redo = False
    w_loop_count = 0
    while pair_count < num_train_pairs + num_test_pairs or redo:
        w_loop_count += 1
        if w_loop_count > 1000:
            return ""
        redo = False

        grid_x_dim = randint(min_grid_dim, max_grid_dim)
        grid_y_dim = randint(min_grid_dim, max_grid_dim)

        input_grid = np.zeros((grid_x_dim, grid_y_dim), dtype=int)
        rand_start_x_index = randint(0, grid_x_dim - 1)
        rand_start_y_index = randint(0, grid_y_dim - 1)

        input_grid[rand_start_x_index, rand_start_y_index] = square_color

        if x_or_y_move > 0:
            moved_x_index = rand_start_x_index + random_move
            moved_y_index = rand_start_y_index
        else:
            moved_x_index = rand_start_x_index
            moved_y_index = rand_start_y_index + random_move

        if (
            moved_x_index < 0
            or moved_y_index < 0
            or moved_x_index > grid_x_dim - 1
            or moved_y_index > grid_y_dim - 1
        ):
            redo = True
        else:
            input_grids.append(input_grid)

            output_grid = np.zeros((grid_x_dim, grid_y_dim), dtype=int)
            output_grid[moved_x_index, moved_y_index] = square_color
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

    # json_string = json.dumps(task)
    json_string = json.dumps(task, cls=NumpyArrayEncoder)

    return json_string


def generate_move_obj_puzzle_med(
    num_train_pairs, num_test_pairs, min_grid_dim, max_grid_dim
):
    input_grids = []
    output_grids = []

    random_move_x = randint(-3, 3)
    random_move_y = randint(-3, 3)

    pair_count = 0
    redo = False
    w_loop_count = 0
    while pair_count < num_train_pairs + num_test_pairs or redo:
        w_loop_count += 1
        if w_loop_count > 1000:
            return ""
        redo = False

        square_color = randint(1, 9)

        grid_x_dim = randint(4, 22)
        grid_y_dim = randint(4, 22)

        input_grid = np.zeros((grid_x_dim, grid_y_dim), dtype=int)
        rand_start_x_index = randint(0, grid_x_dim - 1)
        rand_start_y_index = randint(0, grid_y_dim - 1)

        input_grid[rand_start_x_index, rand_start_y_index] = square_color

        moved_x_index = rand_start_x_index + random_move_x
        moved_y_index = rand_start_y_index + random_move_y

        if (
            moved_x_index < 0
            or moved_y_index < 0
            or moved_x_index > grid_x_dim - 1
            or moved_y_index > grid_y_dim - 1
        ):
            redo = True
        else:
            input_grids.append(input_grid)

            output_grid = np.zeros((grid_x_dim, grid_y_dim), dtype=int)
            output_grid[moved_x_index, moved_y_index] = square_color
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

    # json_string = json.dumps(task)
    json_string = json.dumps(task, cls=NumpyArrayEncoder)

    return json_string


def generate_multiple_obj_puzzle(
    num_train_pairs, num_test_pairs, min_grid_dim, max_grid_dim
):
    sub_grid_x_dim = randint(2, 6)
    sub_grid_y_dim = randint(2, 6)
    size = sub_grid_x_dim * sub_grid_y_dim
    permuted_array = np.random.randint(10, size=size)

    while check_zero_rows_cols(permuted_array, sub_grid_x_dim, sub_grid_y_dim):
        sub_grid_x_dim = randint(2, 6)
        sub_grid_y_dim = randint(2, 6)
        size = sub_grid_x_dim * sub_grid_y_dim
        permuted_array = np.random.randint(10, size=size)

    sub_grid = permuted_array.reshape((sub_grid_x_dim, sub_grid_y_dim))

    input_grids = []
    output_grids = []

    pair_count = 0
    w_loop_count = 0
    while pair_count < num_train_pairs + num_test_pairs:
        w_loop_count += 1
        if w_loop_count > 1000:
            return ""
        grid_x_dim = randint(min_grid_dim, max_grid_dim)
        grid_y_dim = randint(min_grid_dim, max_grid_dim)
        isolated_obj_grid = np.zeros((grid_x_dim, grid_y_dim), dtype=int)

        num_objs = randint(2, 3)
        if (sub_grid_x_dim == 4 and sub_grid_y_dim == 3) or (
            sub_grid_x_dim == 3 and sub_grid_y_dim == 4
        ):
            num_objs = 2

        i = 0
        w2_loop_count = 0
        while i < num_objs:
            w2_loop_count += 1
            if w2_loop_count > 1000:
                return ""
            if i == 0:
                # Place sub-grid randomly inside of output_grid
                rand_start_x_index = randint(0, grid_x_dim - sub_grid_x_dim)
                rand_start_y_index = randint(0, grid_y_dim - sub_grid_y_dim)
                isolated_obj_grid[
                    rand_start_x_index : rand_start_x_index + sub_grid_x_dim,
                    rand_start_y_index : rand_start_y_index + sub_grid_y_dim,
                ] = sub_grid
                i += 1
            else:
                try_again = True
                try_again_count = 0
                w3_loop_count = 0
                while try_again:
                    w3_loop_count += 1
                    if w3_loop_count > 1000:
                        return ""
                    try_again = False
                    # Place sub-grid randomly inside of output_grid
                    rand_start_x_index = randint(0, grid_x_dim - sub_grid_x_dim)
                    rand_start_y_index = randint(0, grid_y_dim - sub_grid_y_dim)
                    candidate_isolated_obj_grid = np.zeros(
                        (grid_x_dim, grid_y_dim), dtype=int
                    )
                    candidate_isolated_obj_grid[
                        rand_start_x_index : rand_start_x_index + sub_grid_x_dim,
                        rand_start_y_index : rand_start_y_index + sub_grid_y_dim,
                    ] = sub_grid
                    break_loops = False
                    for x in range(grid_x_dim):
                        for y in range(grid_y_dim):
                            if (
                                candidate_isolated_obj_grid[x][y] > 0
                                and isolated_obj_grid[x][y] > 0
                            ):
                                break_loops = True
                                try_again = True
                                try_again_count += 1
                                break
                        if break_loops:
                            break

                for x in range(grid_x_dim):
                    for y in range(grid_y_dim):
                        if candidate_isolated_obj_grid[x][y] > 0:
                            isolated_obj_grid[x][y] = candidate_isolated_obj_grid[x][y]
                i += 1

        output_grid = np.copy(isolated_obj_grid)
        output_grids.append(output_grid)
        input_grid = np.copy(output_grid)
        for x in range(grid_x_dim):
            for y in range(grid_y_dim):
                if input_grid[x][y] == 0:
                    input_grid[x][y] = randint(0, 9)
        input_grids.append(input_grid)
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

    json_string = json.dumps(task, cls=NumpyArrayEncoder)

    return json_string


def generate_rays_puzzle(num_train_pairs, num_test_pairs, min_grid_dim, max_grid_dim):
    input_grids = []
    output_grids = []

    rot_num = randint(0, 3)

    pair_count = 0
    w_loop_count = 0
    while pair_count < num_train_pairs + num_test_pairs:
        w_loop_count += 1
        if w_loop_count > 1000:
            return ""
        grid_x_dim = randint(min_grid_dim, max_grid_dim)
        grid_y_dim = randint(min_grid_dim, max_grid_dim)
        color = randint(1, 9)
        input_grid = np.zeros((grid_x_dim, grid_y_dim), dtype=int)
        y_coord = randint(2, grid_y_dim - 2)
        input_grid[0][y_coord] = color

        for x in range(1, grid_x_dim - 1):
            for y in range(1, grid_y_dim - 1):
                loop_again = True
                while loop_again:
                    loop_again = False
                    rand_change = randint(1, 100)
                    if rand_change < 10 and input_grid[x][y] == 0:
                        random_color = randint(1, 9)
                        if x > 1:
                            input_grid[x][y] = random_color
                        else:
                            loop_again = True

        output_grid = np.copy(input_grid)
        for x in range(1, grid_x_dim):
            if output_grid[x][y_coord] == 0:
                output_grid[x][y_coord] = color
            else:
                break

        input_grid = np.rot90(input_grid, k=rot_num)
        input_grids.append(input_grid)
        output_grid = np.rot90(output_grid, k=rot_num)
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
        if i < 3:
            task["train"][i]["input"] = input_grid
            task["train"][i]["output"] = output_grid
        else:
            task["test"][0]["input"] = input_grid
            task["test"][0]["output"] = output_grid
        i += 1

    json_string = json.dumps(task, cls=NumpyArrayEncoder)

    return json_string


def generate_rot_grid_puzzle(
    num_train_pairs, num_test_pairs, min_grid_dim, max_grid_dim
):
    grid_x_dim = randint(min_grid_dim, max_grid_dim)
    grid_y_dim = randint(min_grid_dim, max_grid_dim)

    input_grids = []
    output_grids = []
    rot_num = randint(1, 3)

    pair_count = 0
    w_loop_count = 0
    while pair_count < num_train_pairs + num_test_pairs:
        w_loop_count += 1
        if w_loop_count > 1000:
            return ""
        pair_count += 1
        input_grid = np.zeros((grid_x_dim, grid_y_dim), dtype=int)
        for x in range(grid_x_dim):
            for y in range(grid_y_dim):
                if input_grid[x][y] == 0:
                    input_grid[x][y] = randint(0, 9)
        input_grids.append(input_grid)

        output_grid = np.copy(input_grid)
        output_grid = np.rot90(output_grid, k=rot_num)
        output_grids.append(output_grid)

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

    json_string = json.dumps(task, cls=NumpyArrayEncoder)

    return json_string


def generate_rot_obj_puzzle(
    num_train_pairs, num_test_pairs, min_grid_dim, max_grid_dim
):
    sub_grid_x_dim = randint(2, 6)
    sub_grid_y_dim = randint(2, 6)
    size = sub_grid_x_dim * sub_grid_y_dim
    permuted_array = np.random.randint(10, size=size)

    while check_zero_rows_cols(permuted_array, sub_grid_x_dim, sub_grid_y_dim):
        sub_grid_x_dim = randint(2, 6)
        sub_grid_y_dim = randint(2, 6)
        size = sub_grid_x_dim * sub_grid_y_dim
        permuted_array = np.random.randint(10, size=size)

    sub_grid = permuted_array.reshape((sub_grid_x_dim, sub_grid_y_dim))

    input_grids = []
    base_grids = []
    output_grids = []
    flattened_base_grids = set()

    rot_num = randint(1, 3)
    sub_grid_rotated = np.rot90(sub_grid, k=rot_num)

    pair_count = 0
    w_loop_count = 0
    while pair_count < num_train_pairs + num_test_pairs:
        w_loop_count += 1
        if w_loop_count > 1000:
            return ""
        grid_x_dim = randint(min_grid_dim, max_grid_dim)
        grid_y_dim = randint(min_grid_dim, max_grid_dim)

        base_grid = np.zeros((grid_x_dim, grid_y_dim), dtype=int)
        base_grid_rotated = np.copy(base_grid)

        # Place sub-grid randomly inside of base_grid
        rand_start_x_index = randint(0, grid_x_dim - sub_grid_x_dim)
        rand_start_y_index = randint(0, grid_y_dim - sub_grid_y_dim)
        base_grid[
            rand_start_x_index : rand_start_x_index + sub_grid_x_dim,
            rand_start_y_index : rand_start_y_index + sub_grid_y_dim,
        ] = sub_grid

        flat_tuple = tuple(base_grid.flatten())

        # Add the matrix to the set if it's not already present
        if flat_tuple not in flattened_base_grids:
            if (
                rand_start_x_index + sub_grid_rotated.shape[0] > grid_x_dim - 1
                or rand_start_y_index + sub_grid_rotated.shape[1] > grid_y_dim - 1
            ):
                continue
            else:
                base_grid_rotated[
                    rand_start_x_index : rand_start_x_index + sub_grid_rotated.shape[0],
                    rand_start_y_index : rand_start_y_index + sub_grid_rotated.shape[1],
                ] = sub_grid_rotated

                input_grid = np.copy(base_grid)
                input_grids.append(input_grid)

                output_grid = np.copy(base_grid_rotated)
                output_grids.append(output_grid)

                pair_count += 1

            flattened_base_grids.add(flat_tuple)

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

    json_string = json.dumps(task, cls=NumpyArrayEncoder)

    return json_string


def generate_same_color_puzzle(
    num_train_pairs, num_test_pairs, min_grid_dim, max_grid_dim
):
    input_grids = []
    output_grids = []

    keep_same = randint(0, 1)

    rand_rot_or_flip = randint(0, 5)
    rand_scale_or_shrink = randint(0, 100)

    pair_count = 0
    w_loop_count = 0
    while pair_count < num_train_pairs + num_test_pairs:
        w_loop_count += 1
        if w_loop_count > 1000:
            return ""
        grid_x_dim = randint(min_grid_dim, max_grid_dim)
        grid_y_dim = randint(min_grid_dim, max_grid_dim)

        #################### obj 1 ##################################################
        sub_grid_1_x_dim = randint(2, 2)
        sub_grid_1_y_dim = randint(2, 2)
        permuted_array_1 = np.random.randint(
            10, size=sub_grid_1_x_dim * sub_grid_1_y_dim
        )

        if check_zero_rows_cols(permuted_array_1, sub_grid_1_x_dim, sub_grid_1_y_dim):
            continue

        sub_grid_1 = permuted_array_1.reshape((sub_grid_1_x_dim, sub_grid_1_y_dim))

        sub_grid_1_color = randint(1, 9)
        sub_grid_1_square_count = 0
        for x in range(sub_grid_1_x_dim):
            for y in range(sub_grid_1_y_dim):
                if sub_grid_1[x][y] != 0:
                    sub_grid_1[x][y] = sub_grid_1_color
                    sub_grid_1_square_count += 1

        isolated_obj_grid_1 = np.zeros((grid_x_dim, grid_y_dim), dtype=int)

        start_x_index = randint(0, ((grid_x_dim - (sub_grid_1_x_dim * 2)) // 2))
        start_y_index = randint(0, ((grid_y_dim - (sub_grid_1_y_dim * 2)) // 2))

        isolated_obj_grid_1[
            start_x_index : start_x_index + sub_grid_1_x_dim,
            start_y_index : start_y_index + sub_grid_1_y_dim,
        ] = sub_grid_1

        #################### obj 2 ##################################################
        black_row_or_column = True
        same_square_count = True
        overlapping = True
        w2_loop_count = 0
        while black_row_or_column or same_square_count or overlapping:
            w2_loop_count += 1
            if w2_loop_count > 1000:
                return ""
            black_row_or_column = False
            same_square_count = False
            overlapping = False

            sub_grid_2_x_dim = randint(2, 2)
            sub_grid_2_y_dim = randint(2, 2)

            permuted_array_2 = np.random.randint(
                10, size=sub_grid_2_x_dim * sub_grid_2_y_dim
            )

            if check_zero_rows_cols(
                permuted_array_2, sub_grid_2_x_dim, sub_grid_2_y_dim
            ):
                black_row_or_column = True
                continue

            sub_grid_2 = permuted_array_2.reshape((sub_grid_2_x_dim, sub_grid_2_y_dim))

            sub_grid_2_color = sub_grid_1_color
            sub_grid_2_square_count = 0
            for x in range(sub_grid_2_x_dim):
                for y in range(sub_grid_2_y_dim):
                    if sub_grid_2[x][y] != 0:
                        sub_grid_2[x][y] = sub_grid_2_color
                        sub_grid_2_square_count += 1

            if sub_grid_2_square_count == sub_grid_1_square_count:
                same_square_count = True
                continue

            isolated_obj_grid_2 = np.zeros((grid_x_dim, grid_y_dim), dtype=int)

            start_x_index_2 = randint(0, ((grid_x_dim - (sub_grid_2_x_dim * 2)) // 2))
            start_y_index_2 = randint(0, ((grid_y_dim - (sub_grid_2_y_dim * 2)) // 2))

            isolated_obj_grid_2[
                start_x_index_2 : start_x_index_2 + sub_grid_2_x_dim,
                start_y_index_2 : start_y_index_2 + sub_grid_2_y_dim,
            ] = sub_grid_2

            isolated_obj_grid_2 = np.flipud(isolated_obj_grid_2)
            isolated_obj_grid_2 = np.fliplr(isolated_obj_grid_2)

            break_all = False
            for x in range(grid_x_dim):
                for y in range(grid_y_dim):
                    if (
                        isolated_obj_grid_1[x][y] != 0
                        and isolated_obj_grid_2[x][y] != 0
                    ):
                        overlapping = True
                        break_all = True
                        break
                    elif (
                        x + 1 < grid_x_dim - 1
                        and isolated_obj_grid_1[x][y] != 0
                        and isolated_obj_grid_2[x + 1][y] != 0
                    ):
                        overlapping = True
                        break_all = True
                        break
                    elif (
                        y + 1 < grid_y_dim - 1
                        and isolated_obj_grid_1[x][y] != 0
                        and isolated_obj_grid_2[x][y + 1] != 0
                    ):
                        overlapping = True
                        break_all = True
                        break
                    elif (
                        x + 1 < grid_x_dim - 1
                        and y + 1 < grid_y_dim - 1
                        and isolated_obj_grid_1[x][y] != 0
                        and isolated_obj_grid_2[x + 1][y + 1] != 0
                    ):
                        overlapping = True
                        break_all = True
                        break
                if break_all:
                    break
            if overlapping:
                continue

        #################### obj 3 ##################################################
        black_row_or_column = True
        same_square_count = True
        overlapping = True
        obj_3_loop_count = 0
        w3_loop_count = 0
        while black_row_or_column or same_square_count or overlapping:
            w3_loop_count += 1
            if w3_loop_count > 1000:
                return ""
            obj_3_loop_count += 1
            if obj_3_loop_count > 100:
                print("looping too much", obj_3_loop_count)
            black_row_or_column = False
            same_square_count = False
            overlapping = False

            sub_grid_3_x_dim = randint(2, 2)
            sub_grid_3_y_dim = randint(2, 2)

            permuted_array_3 = np.random.randint(
                10, size=sub_grid_3_x_dim * sub_grid_3_y_dim
            )

            if check_zero_rows_cols(
                permuted_array_3, sub_grid_3_x_dim, sub_grid_3_y_dim
            ):
                black_row_or_column = True
                continue

            sub_grid_3 = permuted_array_3.reshape((sub_grid_3_x_dim, sub_grid_3_y_dim))

            sub_grid_3_color = randint(1, 9)
            while sub_grid_3_color == sub_grid_1_color:
                sub_grid_3_color = randint(1, 9)
            sub_grid_3_square_count = 0
            for x in range(sub_grid_3_x_dim):
                for y in range(sub_grid_3_y_dim):
                    if sub_grid_3[x][y] != 0:
                        sub_grid_3[x][y] = sub_grid_3_color
                        sub_grid_3_square_count += 1

            if sub_grid_3_square_count == sub_grid_1_square_count:
                same_square_count = True
                continue

            isolated_obj_grid_3 = np.zeros((grid_x_dim, grid_y_dim), dtype=int)

            start_x_index_3 = randint(0, ((grid_x_dim - (sub_grid_3_x_dim * 2)) // 2))
            start_y_index_3 = randint(0, ((grid_y_dim - (sub_grid_3_y_dim * 2)) // 2))

            isolated_obj_grid_3[
                start_x_index_3 : start_x_index_3 + sub_grid_3_x_dim,
                start_y_index_3 : start_y_index_3 + sub_grid_3_y_dim,
            ] = sub_grid_3

            isolated_obj_grid_3 = np.flipud(isolated_obj_grid_3)
            # isolated_obj_grid_3 = np.fliplr(isolated_obj_grid_3)

            break_all = False
            for x in range(grid_x_dim):
                for y in range(grid_y_dim):
                    if (
                        isolated_obj_grid_1[x][y] != 0
                        and isolated_obj_grid_3[x][y] != 0
                    ) or (
                        isolated_obj_grid_2[x][y] != 0
                        and isolated_obj_grid_3[x][y] != 0
                    ):
                        overlapping = True
                        break_all = True
                        break
                    elif x + 1 < grid_x_dim - 1 and (
                        (
                            isolated_obj_grid_1[x][y] != 0
                            and isolated_obj_grid_3[x + 1][y] != 0
                        )
                        or (
                            isolated_obj_grid_2[x][y] != 0
                            and isolated_obj_grid_3[x + 1][y] != 0
                        )
                        or (
                            isolated_obj_grid_3[x][y] != 0
                            and isolated_obj_grid_1[x + 1][y] != 0
                        )
                        or (
                            isolated_obj_grid_3[x][y] != 0
                            and isolated_obj_grid_2[x + 1][y] != 0
                        )
                    ):
                        overlapping = True
                        break_all = True
                        break
                    elif y + 1 < grid_y_dim - 1 and (
                        (
                            isolated_obj_grid_1[x][y] != 0
                            and isolated_obj_grid_3[x][y + 1] != 0
                        )
                        or (
                            isolated_obj_grid_2[x][y] != 0
                            and isolated_obj_grid_3[x][y + 1] != 0
                        )
                        or (
                            isolated_obj_grid_3[x][y] != 0
                            and isolated_obj_grid_1[x][y + 1] != 0
                        )
                        or (
                            isolated_obj_grid_3[x][y] != 0
                            and isolated_obj_grid_2[x][y + 1] != 0
                        )
                    ):
                        overlapping = True
                        break_all = True
                        break
                    elif (
                        x + 1 < grid_x_dim - 1
                        and y + 1 < grid_y_dim - 1
                        and (
                            (
                                isolated_obj_grid_1[x][y] != 0
                                and isolated_obj_grid_3[x + 1][y + 1] != 0
                            )
                            or (
                                isolated_obj_grid_2[x][y] != 0
                                and isolated_obj_grid_3[x + 1][y + 1] != 0
                            )
                            or (
                                isolated_obj_grid_3[x][y] != 0
                                and isolated_obj_grid_1[x + 1][y + 1] != 0
                            )
                            or (
                                isolated_obj_grid_3[x][y] != 0
                                and isolated_obj_grid_2[x + 1][y + 1] != 0
                            )
                        )
                    ):
                        overlapping = True
                        break_all = True
                        break
                if break_all:
                    break
            if overlapping:
                continue

        input_grid = np.copy(isolated_obj_grid_1)
        for x in range(grid_x_dim):
            for y in range(grid_y_dim):
                if isolated_obj_grid_2[x][y] != 0:
                    input_grid[x][y] = isolated_obj_grid_2[x][y]
                elif isolated_obj_grid_3[x][y] != 0:
                    input_grid[x][y] = isolated_obj_grid_3[x][y]

        if keep_same > 0:
            output_grid = np.copy(isolated_obj_grid_1)
            for x in range(grid_x_dim):
                for y in range(grid_y_dim):
                    if isolated_obj_grid_2[x][y] != 0:
                        output_grid[x][y] = isolated_obj_grid_2[x][y]
        else:
            output_grid = np.copy(isolated_obj_grid_3)

        #####################################################################################
        if rand_rot_or_flip > 0 and rand_rot_or_flip < 4:
            input_grid = np.rot90(input_grid, k=rand_rot_or_flip)
            output_grid = np.rot90(output_grid, k=rand_rot_or_flip)
        elif rand_rot_or_flip == 4:
            input_grid = np.fliplr(input_grid)
            output_grid = np.fliplr(output_grid)
        elif rand_rot_or_flip == 5:
            input_grid = np.flipud(input_grid)
            output_grid = np.flipud(output_grid)

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

    json_string = json.dumps(task, cls=NumpyArrayEncoder)

    return json_string


def generate_same_shape_puzzle(
    num_train_pairs, num_test_pairs, min_grid_dim, max_grid_dim
):
    input_grids = []
    output_grids = []

    keep_same = randint(0, 1)

    rand_rot_or_flip = randint(0, 5)
    rand_scale_or_shrink = randint(0, 100)

    pair_count = 0
    w_loop_count = 0
    while pair_count < num_train_pairs + num_test_pairs:
        w_loop_count += 1
        if w_loop_count > 1000:
            return ""
        grid_x_dim = randint(min_grid_dim, max_grid_dim)
        grid_y_dim = randint(min_grid_dim, max_grid_dim)

        #################### obj 1 ##################################################
        sub_grid_1_x_dim = randint(2, 3)
        sub_grid_1_y_dim = randint(2, 3)
        permuted_array_1 = np.random.randint(
            10, size=sub_grid_1_x_dim * sub_grid_1_y_dim
        )

        if check_zero_rows_cols(permuted_array_1, sub_grid_1_x_dim, sub_grid_1_y_dim):
            continue

        sub_grid_1 = permuted_array_1.reshape((sub_grid_1_x_dim, sub_grid_1_y_dim))

        sub_grid_1_color = randint(1, 9)
        for x in range(sub_grid_1_x_dim):
            for y in range(sub_grid_1_y_dim):
                if sub_grid_1[x][y] != 0:
                    sub_grid_1[x][y] = sub_grid_1_color

        isolated_obj_grid_1 = np.zeros((grid_x_dim, grid_y_dim), dtype=int)

        start_x_index = randint(0, ((grid_x_dim - (sub_grid_1_x_dim * 2)) // 2))
        start_y_index = randint(0, ((grid_y_dim - (sub_grid_1_y_dim * 2)) // 2))

        isolated_obj_grid_1[
            start_x_index : start_x_index + sub_grid_1_x_dim,
            start_y_index : start_y_index + sub_grid_1_y_dim,
        ] = sub_grid_1

        #################### obj 2 ##################################################
        black_row_or_column = True
        overlapping = True
        w2_loop_count = 0
        while black_row_or_column or overlapping:
            w2_loop_count += 1
            if w2_loop_count > 1000:
                return ""
            black_row_or_column = False
            overlapping = False

            sub_grid_2_x_dim = sub_grid_1_x_dim
            sub_grid_2_y_dim = sub_grid_1_y_dim

            permuted_array_2 = np.random.randint(
                10, size=sub_grid_2_x_dim * sub_grid_2_y_dim
            )

            if check_zero_rows_cols(
                permuted_array_2, sub_grid_2_x_dim, sub_grid_2_y_dim
            ):
                black_row_or_column = True
                continue

            sub_grid_2 = np.copy(sub_grid_1)
            sub_grid_2 = np.flipud(sub_grid_2)
            sub_grid_2 = np.fliplr(sub_grid_2)

            sub_grid_2_color = randint(1, 9)
            while sub_grid_2_color == sub_grid_1_color:
                sub_grid_2_color = randint(1, 9)
            for x in range(sub_grid_2_x_dim):
                for y in range(sub_grid_2_y_dim):
                    if sub_grid_2[x][y] != 0:
                        sub_grid_2[x][y] = sub_grid_2_color

            isolated_obj_grid_2 = np.zeros((grid_x_dim, grid_y_dim), dtype=int)

            start_x_index_2 = randint(0, ((grid_x_dim - (sub_grid_2_x_dim * 2)) // 2))
            start_y_index_2 = randint(0, ((grid_y_dim - (sub_grid_2_y_dim * 2)) // 2))

            isolated_obj_grid_2[
                start_x_index_2 : start_x_index_2 + sub_grid_2_x_dim,
                start_y_index_2 : start_y_index_2 + sub_grid_2_y_dim,
            ] = sub_grid_2

            isolated_obj_grid_2 = np.flipud(isolated_obj_grid_2)
            isolated_obj_grid_2 = np.fliplr(isolated_obj_grid_2)

            break_all = False
            for x in range(grid_x_dim):
                for y in range(grid_y_dim):
                    if (
                        isolated_obj_grid_1[x][y] != 0
                        and isolated_obj_grid_2[x][y] != 0
                    ):
                        overlapping = True
                        break_all = True
                        break
                    elif (
                        x + 1 < grid_x_dim - 1
                        and isolated_obj_grid_1[x][y] != 0
                        and isolated_obj_grid_2[x + 1][y] != 0
                    ):
                        overlapping = True
                        break_all = True
                        break
                    elif (
                        y + 1 < grid_y_dim - 1
                        and isolated_obj_grid_1[x][y] != 0
                        and isolated_obj_grid_2[x][y + 1] != 0
                    ):
                        overlapping = True
                        break_all = True
                        break
                    elif (
                        x + 1 < grid_x_dim - 1
                        and y + 1 < grid_y_dim - 1
                        and isolated_obj_grid_1[x][y] != 0
                        and isolated_obj_grid_2[x + 1][y + 1] != 0
                    ):
                        overlapping = True
                        break_all = True
                        break
                if break_all:
                    break
            if overlapping:
                continue

        #################### obj 3 ##################################################
        black_row_or_column = True
        overlapping = True
        same_as_sub_grid_1 = True
        w4_loop_count = 0
        while black_row_or_column or overlapping or same_as_sub_grid_1:
            w4_loop_count += 1
            if w4_loop_count > 1000:
                return ""
            black_row_or_column = False
            overlapping = False
            same_as_sub_grid_1 = False

            sub_grid_3_x_dim = randint(2, 3)
            sub_grid_3_y_dim = randint(2, 3)

            permuted_array_3 = np.random.randint(
                10, size=sub_grid_3_x_dim * sub_grid_3_y_dim
            )

            if check_zero_rows_cols(
                permuted_array_3, sub_grid_3_x_dim, sub_grid_3_y_dim
            ):
                black_row_or_column = True
                continue

            sub_grid_3 = permuted_array_3.reshape((sub_grid_3_x_dim, sub_grid_3_y_dim))

            sub_grid_3_color = sub_grid_3_color = randint(1, 9)
            while (
                sub_grid_3_color == sub_grid_1_color
                or sub_grid_3_color == sub_grid_2_color
            ):
                sub_grid_3_color = randint(1, 9)
            sub_grid_3_square_count = 0
            for x in range(sub_grid_3_x_dim):
                for y in range(sub_grid_3_y_dim):
                    if sub_grid_3[x][y] != 0:
                        sub_grid_3[x][y] = sub_grid_1_color
                        sub_grid_3_square_count += 1

            if np.array_equal(sub_grid_1, sub_grid_3):
                same_as_sub_grid_1 = True
                continue

            for x in range(sub_grid_3_x_dim):
                for y in range(sub_grid_3_y_dim):
                    if sub_grid_3[x][y] != 0:
                        sub_grid_3[x][y] = sub_grid_3_color

            isolated_obj_grid_3 = np.zeros((grid_x_dim, grid_y_dim), dtype=int)

            start_x_index_3 = randint(0, ((grid_x_dim - (sub_grid_3_x_dim * 2)) // 2))
            start_y_index_3 = randint(0, ((grid_y_dim - (sub_grid_3_y_dim * 2)) // 2))

            isolated_obj_grid_3[
                start_x_index_3 : start_x_index_3 + sub_grid_3_x_dim,
                start_y_index_3 : start_y_index_3 + sub_grid_3_y_dim,
            ] = sub_grid_3

            isolated_obj_grid_3 = np.flipud(isolated_obj_grid_3)
            # isolated_obj_grid_3 = np.fliplr(isolated_obj_grid_3)

            break_all = False
            for x in range(grid_x_dim):
                for y in range(grid_y_dim):
                    if (
                        isolated_obj_grid_1[x][y] != 0
                        and isolated_obj_grid_3[x][y] != 0
                    ) or (
                        isolated_obj_grid_2[x][y] != 0
                        and isolated_obj_grid_3[x][y] != 0
                    ):
                        overlapping = True
                        break_all = True
                        break
                    elif x + 1 < grid_x_dim - 1 and (
                        (
                            isolated_obj_grid_1[x][y] != 0
                            and isolated_obj_grid_3[x + 1][y] != 0
                        )
                        or (
                            isolated_obj_grid_2[x][y] != 0
                            and isolated_obj_grid_3[x + 1][y] != 0
                        )
                        or (
                            isolated_obj_grid_3[x][y] != 0
                            and isolated_obj_grid_1[x + 1][y] != 0
                        )
                        or (
                            isolated_obj_grid_3[x][y] != 0
                            and isolated_obj_grid_2[x + 1][y] != 0
                        )
                    ):
                        overlapping = True
                        break_all = True
                        break

                    elif x - 1 >= 0 and (
                        (
                            isolated_obj_grid_1[x][y] != 0
                            and isolated_obj_grid_3[x - 1][y] != 0
                        )
                        or (
                            isolated_obj_grid_2[x][y] != 0
                            and isolated_obj_grid_3[x - 1][y] != 0
                        )
                        or (
                            isolated_obj_grid_3[x][y] != 0
                            and isolated_obj_grid_1[x - 1][y] != 0
                        )
                        or (
                            isolated_obj_grid_3[x][y] != 0
                            and isolated_obj_grid_2[x - 1][y] != 0
                        )
                    ):
                        overlapping = True
                        break_all = True
                        break

                    elif y + 1 < grid_y_dim - 1 and (
                        (
                            isolated_obj_grid_1[x][y] != 0
                            and isolated_obj_grid_3[x][y + 1] != 0
                        )
                        or (
                            isolated_obj_grid_2[x][y] != 0
                            and isolated_obj_grid_3[x][y + 1] != 0
                        )
                        or (
                            isolated_obj_grid_3[x][y] != 0
                            and isolated_obj_grid_1[x][y + 1] != 0
                        )
                        or (
                            isolated_obj_grid_3[x][y] != 0
                            and isolated_obj_grid_2[x][y + 1] != 0
                        )
                    ):
                        overlapping = True
                        break_all = True
                        break

                    elif y - 1 >= 0 and (
                        (
                            isolated_obj_grid_1[x][y] != 0
                            and isolated_obj_grid_3[x][y - 1] != 0
                        )
                        or (
                            isolated_obj_grid_2[x][y] != 0
                            and isolated_obj_grid_3[x][y - 1] != 0
                        )
                        or (
                            isolated_obj_grid_3[x][y] != 0
                            and isolated_obj_grid_1[x][y - 1] != 0
                        )
                        or (
                            isolated_obj_grid_3[x][y] != 0
                            and isolated_obj_grid_2[x][y - 1] != 0
                        )
                    ):
                        overlapping = True
                        break_all = True
                        break

                    elif (
                        x + 1 < grid_x_dim - 1
                        and y + 1 < grid_y_dim - 1
                        and (
                            (
                                isolated_obj_grid_1[x][y] != 0
                                and isolated_obj_grid_3[x + 1][y + 1] != 0
                            )
                            or (
                                isolated_obj_grid_2[x][y] != 0
                                and isolated_obj_grid_3[x + 1][y + 1] != 0
                            )
                            or (
                                isolated_obj_grid_3[x][y] != 0
                                and isolated_obj_grid_1[x + 1][y + 1] != 0
                            )
                            or (
                                isolated_obj_grid_3[x][y] != 0
                                and isolated_obj_grid_2[x + 1][y + 1] != 0
                            )
                        )
                    ):
                        overlapping = True
                        break_all = True
                        break

                    elif (
                        x - 1 >= 0
                        and y + 1 < grid_y_dim - 1
                        and (
                            (
                                isolated_obj_grid_1[x][y] != 0
                                and isolated_obj_grid_3[x - 1][y + 1] != 0
                            )
                            or (
                                isolated_obj_grid_2[x][y] != 0
                                and isolated_obj_grid_3[x - 1][y + 1] != 0
                            )
                            or (
                                isolated_obj_grid_3[x][y] != 0
                                and isolated_obj_grid_1[x - 1][y + 1] != 0
                            )
                            or (
                                isolated_obj_grid_3[x][y] != 0
                                and isolated_obj_grid_2[x - 1][y + 1] != 0
                            )
                        )
                    ):
                        overlapping = True
                        break_all = True
                        break

                    elif (
                        x + 1 < grid_x_dim - 1
                        and y - 1 >= 0
                        and (
                            (
                                isolated_obj_grid_1[x][y] != 0
                                and isolated_obj_grid_3[x + 1][y - 1] != 0
                            )
                            or (
                                isolated_obj_grid_2[x][y] != 0
                                and isolated_obj_grid_3[x + 1][y - 1] != 0
                            )
                            or (
                                isolated_obj_grid_3[x][y] != 0
                                and isolated_obj_grid_1[x + 1][y - 1] != 0
                            )
                            or (
                                isolated_obj_grid_3[x][y] != 0
                                and isolated_obj_grid_2[x + 1][y - 1] != 0
                            )
                        )
                    ):
                        overlapping = True
                        break_all = True
                        break

                    elif (
                        x - 1 >= 0
                        and y - 1 >= 0
                        and (
                            (
                                isolated_obj_grid_1[x][y] != 0
                                and isolated_obj_grid_3[x - 1][y - 1] != 0
                            )
                            or (
                                isolated_obj_grid_2[x][y] != 0
                                and isolated_obj_grid_3[x - 1][y - 1] != 0
                            )
                            or (
                                isolated_obj_grid_3[x][y] != 0
                                and isolated_obj_grid_1[x - 1][y - 1] != 0
                            )
                            or (
                                isolated_obj_grid_3[x][y] != 0
                                and isolated_obj_grid_2[x - 1][y - 1] != 0
                            )
                        )
                    ):
                        overlapping = True
                        break_all = True
                        break

                if break_all:
                    break
            if overlapping:
                continue

        input_grid = np.copy(isolated_obj_grid_1)
        for x in range(grid_x_dim):
            for y in range(grid_y_dim):
                if isolated_obj_grid_2[x][y] != 0:
                    input_grid[x][y] = isolated_obj_grid_2[x][y]
                elif isolated_obj_grid_3[x][y] != 0:
                    input_grid[x][y] = isolated_obj_grid_3[x][y]

        if keep_same > 0:
            output_grid = np.copy(isolated_obj_grid_1)
            for x in range(grid_x_dim):
                for y in range(grid_y_dim):
                    if isolated_obj_grid_2[x][y] != 0:
                        output_grid[x][y] = isolated_obj_grid_2[x][y]
        else:
            output_grid = np.copy(isolated_obj_grid_3)

        #####################################################################################
        if rand_rot_or_flip > 0 and rand_rot_or_flip < 4:
            input_grid = np.rot90(input_grid, k=rand_rot_or_flip)
            output_grid = np.rot90(output_grid, k=rand_rot_or_flip)
        elif rand_rot_or_flip == 4:
            input_grid = np.fliplr(input_grid)
            output_grid = np.fliplr(output_grid)
        elif rand_rot_or_flip == 5:
            input_grid = np.flipud(input_grid)
            output_grid = np.flipud(output_grid)

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

    json_string = json.dumps(task, cls=NumpyArrayEncoder)

    return json_string


def generate_scale_obj_puzzle_v1(
    num_train_pairs, num_test_pairs, min_grid_dim, max_grid_dim
):
    input_grids = []
    output_grids = []

    rand_rot_or_flip = randint(0, 5)
    rand_scale_or_shrink = randint(0, 100)

    pair_count = 0
    w_loop_count = 0
    while pair_count < num_train_pairs + num_test_pairs:
        w_loop_count += 1
        if w_loop_count > 1000:
            return ""
        grid_x_dim = randint(min_grid_dim, max_grid_dim)
        grid_y_dim = randint(min_grid_dim, max_grid_dim)

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

    json_string = json.dumps(task, cls=NumpyArrayEncoder)

    return json_string


def generate_scale_obj_puzzle_v2(
    num_train_pairs, num_test_pairs, min_grid_dim, max_grid_dim
):
    input_grids = []
    output_grids = []

    rand_rot_or_flip = randint(0, 5)
    rand_scale_or_shrink = randint(0, 100)

    pair_count = 0
    w_loop_count = 0
    while pair_count < num_train_pairs + num_test_pairs:
        w_loop_count += 1
        if w_loop_count > 1000:
            return ""
        grid_x_dim = randint(min_grid_dim, max_grid_dim)
        grid_y_dim = randint(min_grid_dim, max_grid_dim)

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

        if rand_scale_or_shrink > 50:
            input_grid = np.copy(output_grid)
            output_grid = np.copy(sub_grid)
        else:
            output_grid = np.copy(scaled_sub_grid)

        if rand_rot_or_flip > 0 and rand_rot_or_flip < 4:
            input_grid = np.rot90(input_grid, k=rand_rot_or_flip)
            output_grid = np.rot90(output_grid, k=rand_rot_or_flip)
        elif rand_rot_or_flip == 4:
            input_grid = np.fliplr(input_grid)
            output_grid = np.fliplr(output_grid)
        elif rand_rot_or_flip == 5:
            input_grid = np.flipud(input_grid)
            output_grid = np.flipud(output_grid)

        input_grids.append(input_grid)
        output_grids.append(scaled_sub_grid)

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

    json_string = json.dumps(task, cls=NumpyArrayEncoder)

    return json_string


def generate_swap_colors_puzzle(
    num_train_pairs, num_test_pairs, min_grid_dim, max_grid_dim
):
    input_grids = []
    output_grids = []

    colors = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    permuted_colors = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    color_perm_num = randint(1, 9)
    colors_used_in_train_tasks = []

    pair_count = 0
    w_loop_count = 0
    while pair_count < num_train_pairs + num_test_pairs:
        w_loop_count += 1
        if w_loop_count > 1000:
            return ""
        grid_x_dim = randint(min_grid_dim, max_grid_dim)
        grid_y_dim = randint(min_grid_dim, max_grid_dim)

        if pair_count < 3:
            input_grid = np.random.randint(0, 9, size=(grid_x_dim * grid_y_dim))
            for c in input_grid:
                if c not in colors_used_in_train_tasks:
                    colors_used_in_train_tasks.append(c)
        else:
            # Generate a random length for the array
            length = grid_x_dim * grid_y_dim
            # Generate the random array
            input_grid = np.random.choice(colors_used_in_train_tasks, size=length)

        input_grid = np.reshape(input_grid, (grid_x_dim, grid_y_dim))

        output_grid = np.copy(input_grid)
        for perm_loop in range(len(colors)):
            swap_index = perm_loop + color_perm_num
            if swap_index >= len(colors):
                swap_index = (perm_loop + color_perm_num) - len(colors)
            permuted_colors[perm_loop] = colors[swap_index]
        mapping = dict(zip(colors, permuted_colors))
        for x in range(grid_x_dim):
            for y in range(grid_y_dim):
                output_grid[x, y] = mapping[output_grid[x, y]]

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
        if i < 3:
            task["train"][i]["input"] = input_grid
            task["train"][i]["output"] = output_grid
        else:
            task["test"][0]["input"] = input_grid
            task["test"][0]["output"] = output_grid
        i += 1

    json_string = json.dumps(task, cls=NumpyArrayEncoder)

    return json_string


def generate_plus_puzzle(num_train_pairs, num_test_pairs, min_grid_dim, max_grid_dim):
    input_grids = []
    output_grids = []

    random_extension_length = randint(1, 5)

    pair_count = 0
    w_loop_count = 0
    while pair_count < num_train_pairs + num_test_pairs:
        w_loop_count += 1
        if w_loop_count > 1000:
            return ""
        grid_x_dim = randint(min_grid_dim, max_grid_dim)
        grid_y_dim = randint(min_grid_dim, max_grid_dim)

        isolated_obj_grid = np.zeros((grid_x_dim, grid_y_dim), dtype=int)

        sub_grid_x_dim = randint(2, 6)
        sub_grid_y_dim = randint(2, 6)
        size = sub_grid_x_dim * sub_grid_y_dim
        permuted_array = np.random.randint(1, 10, size=size)

        sub_grid = permuted_array.reshape((sub_grid_x_dim, sub_grid_y_dim))

        # Place sub-grid randomly inside of isolated_obj_grid
        rand_start_x_index = randint(0, grid_x_dim - sub_grid_x_dim)
        rand_start_y_index = randint(0, grid_y_dim - sub_grid_y_dim)
        isolated_obj_grid[
            rand_start_x_index : rand_start_x_index + sub_grid_x_dim,
            rand_start_y_index : rand_start_y_index + sub_grid_y_dim,
        ] = sub_grid

        input_grid = np.copy(isolated_obj_grid)

        # output_grid extended the edge squares of the sub-grid by random_extension_length
        output_grid = np.copy(isolated_obj_grid)
        # Find the edge squares of the sub-grid
        edge_squares = []
        for x in range(grid_x_dim):
            for y in range(grid_y_dim):
                if (output_grid[x, y] != 0) and (
                    x == rand_start_x_index
                    or x == rand_start_x_index + sub_grid_x_dim - 1
                    or y == rand_start_y_index
                    or y == rand_start_y_index + sub_grid_y_dim - 1
                ):
                    edge_squares.append([x, y])

        # Extend the edge squares up/down/right/left/diagonally by random_extension_length
        for square in edge_squares:
            x, y = square
            # above
            if x - 1 > -1 and output_grid[x - 1, y] == 0:
                new_x = x - 1
                while new_x > -1 and output_grid[new_x, y] == 0:
                    output_grid[new_x, y] = output_grid[x, y]
                    new_x = new_x - 1
                    x_dist = abs(new_x - x)
                    if x_dist > random_extension_length:
                        break

            # below
            if x + 1 < grid_x_dim and output_grid[x + 1, y] == 0:
                new_x = x + 1
                while new_x < grid_x_dim and output_grid[new_x, y] == 0:
                    output_grid[new_x, y] = output_grid[x, y]
                    new_x = new_x + 1
                    x_dist = abs(new_x - x)
                    if x_dist > random_extension_length:
                        break

            # left
            if y - 1 > -1 and output_grid[x, y - 1] == 0:
                new_y = y - 1
                while new_y > -1 and output_grid[x, new_y] == 0:
                    output_grid[x, new_y] = output_grid[x, y]
                    new_y = new_y - 1
                    y_dist = abs(new_y - y)
                    if y_dist > random_extension_length:
                        break

            # right
            if y + 1 < grid_y_dim and output_grid[x, y + 1] == 0:
                new_y = y + 1
                while new_y < grid_y_dim and output_grid[x, new_y] == 0:
                    output_grid[x, new_y] = output_grid[x, y]
                    new_y = new_y + 1
                    y_dist = abs(new_y - y)
                    if y_dist > random_extension_length:
                        break

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

    json_string = json.dumps(task, cls=NumpyArrayEncoder)

    return json_string


def generate_x_puzzle(num_train_pairs, num_test_pairs, min_grid_dim, max_grid_dim):
    input_grids = []
    output_grids = []

    random_extension_length = randint(1, 5)

    pair_count = 0
    w_loop_count = 0
    while pair_count < num_train_pairs + num_test_pairs:
        w_loop_count += 1
        if w_loop_count > 1000:
            return ""
        grid_x_dim = randint(min_grid_dim, max_grid_dim)
        grid_y_dim = randint(min_grid_dim, max_grid_dim)

        isolated_obj_grid = np.zeros((grid_x_dim, grid_y_dim), dtype=int)

        sub_grid_x_dim = randint(2, 6)
        sub_grid_y_dim = randint(2, 6)
        size = sub_grid_x_dim * sub_grid_y_dim
        permuted_array = np.random.randint(1, 10, size=size)

        sub_grid = permuted_array.reshape((sub_grid_x_dim, sub_grid_y_dim))

        # Place sub-grid randomly inside of isolated_obj_grid
        rand_start_x_index = randint(0, grid_x_dim - sub_grid_x_dim)
        rand_start_y_index = randint(0, grid_y_dim - sub_grid_y_dim)
        isolated_obj_grid[
            rand_start_x_index : rand_start_x_index + sub_grid_x_dim,
            rand_start_y_index : rand_start_y_index + sub_grid_y_dim,
        ] = sub_grid

        input_grid = np.copy(isolated_obj_grid)

        # output_grid extended the edge squares of the sub-grid by random_extension_length
        output_grid = np.copy(isolated_obj_grid)
        # Find the edge squares of the sub-grid
        edge_squares = []
        for x in range(grid_x_dim):
            for y in range(grid_y_dim):
                if (output_grid[x, y] != 0) and (
                    x == rand_start_x_index
                    or x == rand_start_x_index + sub_grid_x_dim - 1
                    or y == rand_start_y_index
                    or y == rand_start_y_index + sub_grid_y_dim - 1
                ):
                    edge_squares.append([x, y])

        # Extend the edge squares up/down/right/left/diagonally by random_extension_length
        for square in edge_squares:
            x, y = square

            # diagonally up and left
            if (
                x - 1 > -1
                and output_grid[x - 1, y] == 0
                and y - 1 > -1
                and output_grid[x, y - 1] == 0
            ):
                new_x = x - 1
                new_y = y - 1
                while new_x > -1 and new_y > -1 and output_grid[new_x, new_y] == 0:
                    output_grid[new_x, new_y] = output_grid[x, y]
                    new_x = new_x - 1
                    new_y = new_y - 1
                    x_dist = abs(new_x - x)
                    if x_dist > random_extension_length:
                        break

            # diagonally up and right
            if (
                x - 1 > -1
                and output_grid[x - 1, y] == 0
                and y + 1 < grid_y_dim
                and output_grid[x, y + 1] == 0
            ):
                new_x = x - 1
                new_y = y + 1
                while (
                    new_x > -1 and new_y < grid_y_dim and output_grid[new_x, new_y] == 0
                ):
                    output_grid[new_x, new_y] = output_grid[x, y]
                    new_x = new_x - 1
                    new_y = new_y + 1
                    x_dist = abs(new_x - x)
                    if x_dist > random_extension_length:
                        break

            # diagonally down and left
            if (
                x + 1 < grid_x_dim
                and output_grid[x + 1, y] == 0
                and y - 1 > -1
                and output_grid[x, y - 1] == 0
            ):
                new_x = x + 1
                new_y = y - 1
                while (
                    new_x < grid_x_dim and new_y > -1 and output_grid[new_x, new_y] == 0
                ):
                    output_grid[new_x, new_y] = output_grid[x, y]
                    new_x = new_x + 1
                    new_y = new_y - 1
                    x_dist = abs(new_x - x)
                    if x_dist > random_extension_length:
                        break
            # diagonally down and right
            if (
                x + 1 < grid_x_dim
                and output_grid[x + 1, y] == 0
                and y + 1 < grid_y_dim
                and output_grid[x, y + 1] == 0
            ):
                new_x = x + 1
                new_y = y + 1
                while (
                    new_x < grid_x_dim
                    and new_y < grid_y_dim
                    and output_grid[new_x, new_y] == 0
                ):
                    output_grid[new_x, new_y] = output_grid[x, y]
                    new_x = new_x + 1
                    new_y = new_y + 1
                    x_dist = abs(new_x - x)
                    if x_dist > random_extension_length:
                        break

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

    json_string = json.dumps(task, cls=NumpyArrayEncoder)

    return json_string


############################################################


def generate_random_puzzle_step_1(
    num_train_pairs, num_test_pairs, min_grid_dim, max_grid_dim
):
    puzzle_generators = [
        generate_output_grid_size_equals_input_puzzle_easy,
    ]
    num_puzzle_generators = len(puzzle_generators)

    random_puzzle_index = randint(0, num_puzzle_generators - 1)
    random_puzzle_json = ""
    while random_puzzle_json == "":
        random_puzzle_json = puzzle_generators[random_puzzle_index](
            num_train_pairs, num_test_pairs, min_grid_dim, max_grid_dim
        )

    return random_puzzle_json


def generate_random_puzzle_step_2(
    num_train_pairs, num_test_pairs, min_grid_dim, max_grid_dim
):
    puzzle_generators = [
        generate_output_grid_size_equals_input_puzzle,
    ]
    num_puzzle_generators = len(puzzle_generators)

    random_puzzle_index = randint(0, num_puzzle_generators - 1)
    random_puzzle_json = ""
    while random_puzzle_json == "":
        random_puzzle_json = puzzle_generators[random_puzzle_index](
            num_train_pairs, num_test_pairs, min_grid_dim, max_grid_dim
        )

    return random_puzzle_json


def generate_random_puzzle_step_3(
    num_train_pairs, num_test_pairs, min_grid_dim, max_grid_dim
):
    puzzle_generators = [
        generate_move_obj_puzzle_super_easy,
    ]
    num_puzzle_generators = len(puzzle_generators)

    random_puzzle_index = randint(0, num_puzzle_generators - 1)
    random_puzzle_json = ""
    while random_puzzle_json == "":
        random_puzzle_json = puzzle_generators[random_puzzle_index](
            num_train_pairs, num_test_pairs, min_grid_dim, max_grid_dim
        )

    return random_puzzle_json


def generate_random_puzzle_step_4(
    num_train_pairs, num_test_pairs, min_grid_dim, max_grid_dim
):
    puzzle_generators = [
        generate_move_obj_puzzle_easy,
    ]
    num_puzzle_generators = len(puzzle_generators)

    random_puzzle_index = randint(0, num_puzzle_generators - 1)
    random_puzzle_json = ""
    while random_puzzle_json == "":
        random_puzzle_json = puzzle_generators[random_puzzle_index](
            num_train_pairs, num_test_pairs, min_grid_dim, max_grid_dim
        )

    return random_puzzle_json


def generate_random_puzzle_step_5(
    num_train_pairs, num_test_pairs, min_grid_dim, max_grid_dim
):
    puzzle_generators = [
        generate_move_obj_puzzle_med,
    ]
    num_puzzle_generators = len(puzzle_generators)

    random_puzzle_index = randint(0, num_puzzle_generators - 1)
    random_puzzle_json = ""
    while random_puzzle_json == "":
        random_puzzle_json = puzzle_generators[random_puzzle_index](
            num_train_pairs, num_test_pairs, min_grid_dim, max_grid_dim
        )

    return random_puzzle_json


def generate_random_puzzle_step_6(
    num_train_pairs, num_test_pairs, min_grid_dim, max_grid_dim
):
    puzzle_generators = [
        generate_move_obj_puzzle,
    ]
    num_puzzle_generators = len(puzzle_generators)

    random_puzzle_index = randint(0, num_puzzle_generators - 1)
    random_puzzle_json = ""
    while random_puzzle_json == "":
        random_puzzle_json = puzzle_generators[random_puzzle_index](
            num_train_pairs, num_test_pairs, min_grid_dim, max_grid_dim
        )

    return random_puzzle_json


def generate_random_puzzle_step_7(
    num_train_pairs, num_test_pairs, min_grid_dim, max_grid_dim
):
    puzzle_generators = [
        generate_move_obj_puzzle,
        generate_isolate_obj_puzzle_easy,
    ]
    num_puzzle_generators = len(puzzle_generators)

    random_puzzle_index = randint(0, num_puzzle_generators - 1)
    random_puzzle_json = ""
    while random_puzzle_json == "":
        random_puzzle_json = puzzle_generators[random_puzzle_index](
            num_train_pairs, num_test_pairs, min_grid_dim, max_grid_dim
        )

    return random_puzzle_json


def generate_random_puzzle_step_8(
    num_train_pairs, num_test_pairs, min_grid_dim, max_grid_dim
):
    puzzle_generators = [
        generate_move_obj_puzzle,
        generate_isolate_obj_puzzle_med,
    ]
    num_puzzle_generators = len(puzzle_generators)

    random_puzzle_index = randint(0, num_puzzle_generators - 1)
    random_puzzle_json = ""
    while random_puzzle_json == "":
        random_puzzle_json = puzzle_generators[random_puzzle_index](
            num_train_pairs, num_test_pairs, min_grid_dim, max_grid_dim
        )

    return random_puzzle_json


def generate_random_puzzle_step_9(
    num_train_pairs, num_test_pairs, min_grid_dim, max_grid_dim
):
    puzzle_generators = [
        generate_move_obj_puzzle,
        generate_isolate_obj_puzzle,
    ]
    num_puzzle_generators = len(puzzle_generators)

    random_puzzle_index = randint(0, num_puzzle_generators - 1)
    random_puzzle_json = ""
    while random_puzzle_json == "":
        random_puzzle_json = puzzle_generators[random_puzzle_index](
            num_train_pairs, num_test_pairs, min_grid_dim, max_grid_dim
        )

    return random_puzzle_json


def generate_random_puzzle(num_train_pairs, num_test_pairs, min_grid_dim, max_grid_dim):
    puzzle_generators = [
        generate_output_grid_size_equals_input_puzzle_easy,
        generate_output_grid_size_equals_input_puzzle,
        generate_move_obj_puzzle_super_easy,
        generate_move_obj_puzzle_easy,
        generate_move_obj_puzzle_med,
        generate_move_obj_puzzle,
        generate_isolate_obj_puzzle_easy,
        generate_isolate_obj_puzzle_med,
        generate_isolate_obj_puzzle,
        generate_count_squares_puzzle,
        generate_fill_holes_in_line_patterns_puzzle,
        generate_fill_holes_in_pattern_puzzle_v1,
        generate_fill_holes_in_pattern_puzzle_v2,
        generate_fill_surrounded_puzzle,
        generate_gravity_puzzle,
        generate_largest_smallest_obj_puzzle_v1,
        generate_largest_smallest_obj_puzzle_v2,
        generate_mask_puzzle,
        generate_mirror_grid_puzzle,
        generate_mirror_obj_puzzle,
        generate_most_freq_in_row_puzzle,
        generate_multiple_obj_puzzle,
        generate_rays_puzzle,
        generate_rot_grid_puzzle,
        generate_same_color_puzzle,
        generate_same_shape_puzzle,
        generate_scale_obj_puzzle_v1,
        generate_scale_obj_puzzle_v2,
        generate_swap_colors_puzzle,
        generate_plus_puzzle,
        generate_x_puzzle,
    ]
    num_puzzle_generators = len(puzzle_generators)

    random_puzzle_index = randint(0, num_puzzle_generators - 1)
    random_puzzle_json = ""
    while random_puzzle_json == "":
        random_puzzle_json = puzzle_generators[random_puzzle_index](
            num_train_pairs, num_test_pairs, min_grid_dim, max_grid_dim
        )

    return random_puzzle_json


def generate_random_core_puzzle(
    num_train_pairs, num_test_pairs, min_grid_dim, max_grid_dim
):
    puzzle_generators = [
        generate_move_obj_puzzle,
        generate_isolate_obj_puzzle,
        generate_count_squares_puzzle,
        generate_fill_holes_in_line_patterns_puzzle,
        generate_fill_holes_in_pattern_puzzle_v1,
        generate_fill_holes_in_pattern_puzzle_v2,
        generate_fill_surrounded_puzzle,
        generate_gravity_puzzle,
        generate_largest_smallest_obj_puzzle_v1,
        generate_largest_smallest_obj_puzzle_v2,
        generate_mask_puzzle,
        generate_mirror_grid_puzzle,
        generate_mirror_obj_puzzle,
        generate_most_freq_in_row_puzzle,
        generate_multiple_obj_puzzle,
        generate_rays_puzzle,
        generate_rot_grid_puzzle,
        generate_same_color_puzzle,
        generate_same_shape_puzzle,
        generate_scale_obj_puzzle_v1,
        generate_scale_obj_puzzle_v2,
        generate_swap_colors_puzzle,
        generate_plus_puzzle,
        generate_x_puzzle,
    ]
    num_puzzle_generators = len(puzzle_generators)

    random_puzzle_index = randint(0, num_puzzle_generators - 1)
    random_puzzle_json = ""
    while random_puzzle_json == "":
        random_puzzle_json = puzzle_generators[random_puzzle_index](
            num_train_pairs, num_test_pairs, min_grid_dim, max_grid_dim
        )

    return random_puzzle_json


############################################################


def make_json_file(base_file_name, json_string, data_folder):
    augmented_filename = base_file_name + ".json"
    if not os.path.exists(data_folder):
        os.mkdir(data_folder)

    filepath = data_folder + augmented_filename
    with open(filepath, "w") as outfile:
        outfile.write(json_string)


############################################################


# def plot_task(task):
#     """
#     Plots all train and test pairs of a specified task,
#     using same color scheme as the ARC app
#     """
#     # cmap = plt.cm.tab10
#     cmap = matplotlib.colors.ListedColormap(
#         [
#             "#000000",
#             "#0074D9",
#             "#FF4136",
#             "#2ECC40",
#             "#FFDC00",
#             "#AAAAAA",
#             "#F012BE",
#             "#FF851B",
#             "#7FDBFF",
#             "#870C25",
#         ]
#     )
#     norm = plt.Normalize(vmin=0, vmax=9)

#     n_train = len(task["train"])
#     n_test = len(task["test"])

#     fig, axs = plt.subplots(n_train + n_test, 2, figsize=(4, 2 * (n_train + n_test)))

#     for i in range(n_train):
#         axs[i, 0].imshow(task["train"][i]["input"], cmap=cmap, norm=norm)
#         axs[i, 0].axis("off")
#         axs[i, 0].set_title("Train Input {}".format(i + 1))
#         axs[i, 1].imshow(task["train"][i]["output"], cmap=cmap, norm=norm)
#         axs[i, 1].axis("off")
#         axs[i, 1].set_title("Train Output {}".format(i + 1))

#     for i in range(n_test):
#         axs[i + n_train, 0].imshow(task["test"][i]["input"], cmap=cmap, norm=norm)
#         axs[i + n_train, 0].axis("off")
#         axs[i + n_train, 0].set_title("Test Input {}".format(i + 1))
#         axs[i + n_train, 1].imshow(task["test"][i]["output"], cmap=cmap, norm=norm)
#         axs[i + n_train, 1].axis("off")
#         axs[i + n_train, 1].set_title("Test Output {}".format(i + 1))

#     # plt.tight_layout()
#     plt.show()


############################################################


def format_instruction_and_output(puzzle_string):
    puzzle_dict = json.loads(puzzle_string)
    train_pairs = puzzle_dict["train"]
    test_pairs = puzzle_dict["test"]

    instruction = ""
    output = ""

    for i, train_pair in enumerate(train_pairs):
        if i > 0:
            instruction += " "
        instruction += "Train_" + str(i + 1) + "_Input = " + str(train_pair["input"])
        instruction += " Train_" + str(i + 1) + "_Output = " + str(train_pair["output"])

    for i, test_pair in enumerate(test_pairs):
        instruction += " Test_" + str(i + 1) + "_Input = " + str(test_pair["input"])
        output += "Test_" + str(i + 1) + "_Output = " + str(test_pair["output"])

    return instruction, output


############################################################


if __name__ == "__main__":
    # num_train_pairs = 3
    # num_test_pairs = 1
    # min_grid_dim = 8
    # max_grid_dim = 30

    # random_puzzle_json = generate_x_puzzle(
    #     num_train_pairs, num_test_pairs, min_grid_dim, max_grid_dim
    # )
    # plot_task(json.loads(random_puzzle_json))

    # num_puzzles = 10
    # for i in range(num_puzzles):
    #     random_puzzle_json = generate_random_core_puzzle(
    #         num_train_pairs, num_test_pairs, min_grid_dim, max_grid_dim
    #     )
    #     base_file_name = "core_puzzle_" + str(i)
    #     make_json_file(base_file_name, random_puzzle_json, "data/core_puzzles/")

    puzzles = []
    num_step_1a_puzzles = 1000
    for i in range(num_step_1a_puzzles):
        num_train_pairs = 3
        num_test_pairs = 1
        min_grid_dim = 4
        max_grid_dim = 6
        random_puzzle_json = generate_random_puzzle_step_1(
            num_train_pairs, num_test_pairs, min_grid_dim, max_grid_dim
        )
        instruction, output = format_instruction_and_output(random_puzzle_json)
        puzzles.append({"instruction": instruction, "output": output})

    json_string = json.dumps(puzzles, cls=NumpyArrayEncoder)
    base_file_name = "data/ARCSolver_training_puzzles"
    filename = base_file_name + ".json"
    filepath = filename
    with open(filepath, "w") as outfile:
        outfile.write(json_string)
