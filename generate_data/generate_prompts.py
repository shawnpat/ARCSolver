from math import perm
import attr
import numpy as np
from sympy import per
from torch import fill
from transformers import AutoTokenizer
from random import randint


def check_zero_rows_cols(permuted_array, sub_grid_row_dim, sub_grid_col_dim):
    reshaped_array = permuted_array.reshape((sub_grid_row_dim, sub_grid_col_dim))
    return np.any(np.all(reshaped_array == 0, axis=0)) or np.any(
        np.all(reshaped_array == 0, axis=1)
    )


def create_sub_grid_and_place_in_base(
    grid_row_dim,
    grid_col_dim,
    sub_grid_row_dim,
    sub_grid_col_dim,
    permuted_array,
    flip_type,
):
    base_grid = np.zeros((grid_row_dim, grid_col_dim), dtype=int)
    base_grid_mirrored = np.zeros((grid_row_dim, grid_col_dim), dtype=int)

    sub_grid = permuted_array.reshape((sub_grid_row_dim, sub_grid_col_dim))
    sub_grid_mirrored = np.flipud(sub_grid) if flip_type == 0 else np.fliplr(sub_grid)

    # Place sub-grid randomly inside of base_grid
    rand_start_row_index = randint(0, grid_row_dim - sub_grid_row_dim)
    rand_start_col_index = randint(0, grid_col_dim - sub_grid_col_dim)
    base_grid[
        rand_start_row_index : rand_start_row_index + sub_grid_row_dim,
        rand_start_col_index : rand_start_col_index + sub_grid_col_dim,
    ] = sub_grid
    base_grid_mirrored[
        rand_start_row_index : rand_start_row_index + sub_grid_row_dim,
        rand_start_col_index : rand_start_col_index + sub_grid_col_dim,
    ] = sub_grid_mirrored

    return base_grid, base_grid_mirrored


def generate_random_move(min_move=-3, max_move=3):
    random_move_row = randint(min_move, max_move)
    random_move_col = randint(min_move, max_move)
    while random_move_row == 0 and random_move_col == 0:
        random_move_col = randint(min_move, max_move)
    return random_move_row, random_move_col


def make_move_obj_grids(
    min_grid_dim, max_grid_dim, num_train_tasks, num_test_tasks, max_attempts=1000
):
    input_grids = []
    output_grids = []

    random_move_row, random_move_col = generate_random_move()

    pair_count = 0
    attempts = 0
    while pair_count < num_train_tasks + num_test_tasks and attempts < max_attempts:
        attempts += 1

        grid_row_dim = randint(min_grid_dim, max_grid_dim)
        grid_col_dim = randint(min_grid_dim, max_grid_dim)

        min_sub_grid_dim = 2
        max_sub_grid_dim = 5
        sub_grid_row_dim = randint(min_sub_grid_dim, max_sub_grid_dim)
        sub_grid_col_dim = randint(min_sub_grid_dim, max_sub_grid_dim)

        size = sub_grid_row_dim * sub_grid_col_dim
        permuted_array = np.random.randint(10, size=size)

        if check_zero_rows_cols(permuted_array, sub_grid_row_dim, sub_grid_col_dim):
            continue

        base_grid = np.zeros((grid_row_dim, grid_col_dim), dtype=int)

        # Place sub-grid randomly inside of base_grid
        sub_grid = permuted_array.reshape((sub_grid_row_dim, sub_grid_col_dim))
        rand_start_row_index = randint(0, grid_row_dim - sub_grid_row_dim)
        rand_start_col_index = randint(0, grid_col_dim - sub_grid_col_dim)
        base_grid[
            rand_start_row_index : rand_start_row_index + sub_grid_row_dim,
            rand_start_col_index : rand_start_col_index + sub_grid_col_dim,
        ] = sub_grid

        # Move the sub-grid within the base grid
        start_row = rand_start_row_index + random_move_row
        start_col = rand_start_col_index + random_move_col
        if (
            start_row < 0
            or start_col < 0
            or start_row + sub_grid_row_dim > grid_row_dim
            or start_col + sub_grid_col_dim > grid_col_dim
        ):
            continue

        moved_obj_grid = np.zeros((grid_row_dim, grid_col_dim), dtype=int)
        moved_obj_grid[
            start_row : start_row + sub_grid_row_dim,
            start_col : start_col + sub_grid_col_dim,
        ] = sub_grid

        input_grids.append(base_grid)
        output_grids.append(moved_obj_grid)
        pair_count += 1

    if attempts == max_attempts:
        print("make_move_obj_grids: Failed to create grids after maximum attempts")
        return None, None, None, None

    return random_move_row, random_move_col, input_grids, output_grids


def make_rotate_obj_grids(
    min_grid_dim, max_grid_dim, num_train_tasks, num_test_tasks, max_attempts=1000
):
    input_grids = []
    output_grids = []

    rand_rot_grid = randint(0, 3)
    rand_flip_grid = randint(0, 1)
    rot_num = randint(1, 3)

    attempts = 0
    pair_count = 0
    check_zero_rows_cols_loops = 0
    loops = 0
    while pair_count < num_train_tasks + num_test_tasks and attempts < max_attempts:
        attempts += 1

        grid_row_dim = randint(min_grid_dim, max_grid_dim)
        grid_col_dim = randint(min_grid_dim, max_grid_dim)

        min_sub_grid_dim = 2
        max_sub_grid_dim = 5
        sub_grid_row_dim = randint(min_sub_grid_dim, max_sub_grid_dim)
        sub_grid_col_dim = randint(min_sub_grid_dim, max_sub_grid_dim)

        size = sub_grid_row_dim * sub_grid_col_dim
        permuted_array = np.random.randint(10, size=size)

        check_zero_rows_cols_loops = 0
        if check_zero_rows_cols(permuted_array, sub_grid_row_dim, sub_grid_col_dim):
            check_zero_rows_cols_loops += 1
            continue

        sub_grid = permuted_array.reshape((sub_grid_row_dim, sub_grid_col_dim))

        # Place sub-grid randomly inside of base_grid
        base_grid = np.zeros((grid_row_dim, grid_col_dim), dtype=int)
        input_grid = np.copy(base_grid)

        rand_start_row_index = randint(0, grid_row_dim - sub_grid_row_dim)
        rand_start_col_index = randint(0, grid_col_dim - sub_grid_col_dim)
        max_loops = 1000
        loops = 0
        while (
            rand_start_row_index + sub_grid_row_dim > grid_row_dim
            or rand_start_col_index + sub_grid_col_dim > grid_col_dim
            and loops < max_loops
        ):
            loops += 1
            rand_start_row_index = randint(0, grid_row_dim - sub_grid_row_dim)
            rand_start_col_index = randint(0, grid_col_dim - sub_grid_col_dim)

        if loops >= max_loops:
            continue

        input_grid[
            rand_start_row_index : rand_start_row_index + sub_grid_row_dim,
            rand_start_col_index : rand_start_col_index + sub_grid_col_dim,
        ] = sub_grid

        output_grid = np.copy(base_grid)

        # Rotate the sub-grid
        sub_grid_rotated = np.rot90(sub_grid, rot_num)

        # Get the new starting position of the rotated sub-grid if it is rotated around the initial top-left corner
        if rot_num == 1:  # 90 degrees counter-clockwise
            new_start_row = rand_start_row_index - (sub_grid_rotated.shape[0] - 1)
            new_start_col = rand_start_col_index
        elif rot_num == 2:  # 180 degrees counter-clockwise
            new_start_row = rand_start_row_index - (sub_grid_rotated.shape[0] - 1)
            new_start_col = rand_start_col_index - (sub_grid_rotated.shape[1] - 1)
        else:  # 270 degrees counter-clockwise
            new_start_row = rand_start_row_index
            new_start_col = rand_start_col_index - (sub_grid_rotated.shape[1] - 1)

        for i in range(new_start_row, new_start_row + sub_grid_rotated.shape[0]):
            for j in range(new_start_col, new_start_col + sub_grid_rotated.shape[1]):
                if i < 0 or j < 0 or i >= grid_row_dim or j >= grid_col_dim:
                    continue
                output_grid[i][j] = sub_grid_rotated[i - new_start_row][
                    j - new_start_col
                ]

        if rand_rot_grid > 0:
            input_grid = np.rot90(input_grid, k=rand_rot_grid)
            output_grid = np.rot90(output_grid, k=rand_rot_grid)

        if rand_flip_grid > 0:
            input_grid = np.fliplr(input_grid)
            output_grid = np.fliplr(output_grid)

        input_grids.append(np.copy(input_grid))
        output_grids.append(np.copy(output_grid))

        pair_count += 1

    if attempts == max_attempts:
        print("make_rotate_obj_grids: Failed to create grids after maximum attempts")
        print("check_zero_rows_cols_loops", check_zero_rows_cols_loops)
        print("loops", loops)
        print("attempts", attempts)
        return None, None, None, None, None

    return rand_rot_grid, rand_flip_grid, rot_num, input_grids, output_grids


def make_mirrored_obj_grids(
    min_grid_dim, max_grid_dim, num_train_tasks, num_test_tasks, max_attempts=1000
):
    input_grids = []
    output_grids = []

    min_sub_grid_dim = 2
    max_sub_grid_dim = 5

    flip_type = randint(0, 1)

    attempts = 0
    pair_count = 0
    while pair_count < num_train_tasks + num_test_tasks:
        attempts += 1

        grid_row_dim = randint(min_grid_dim, max_grid_dim)
        grid_col_dim = randint(min_grid_dim, max_grid_dim)

        sub_grid_row_dim = randint(min_sub_grid_dim, max_sub_grid_dim)
        sub_grid_col_dim = randint(min_sub_grid_dim, max_sub_grid_dim)

        size = sub_grid_row_dim * sub_grid_col_dim
        permuted_array = np.random.randint(10, size=size)

        if check_zero_rows_cols(permuted_array, sub_grid_row_dim, sub_grid_col_dim):
            continue

        base_grid, base_grid_mirrored = create_sub_grid_and_place_in_base(
            grid_row_dim,
            grid_col_dim,
            sub_grid_row_dim,
            sub_grid_col_dim,
            permuted_array,
            flip_type,
        )

        input_grids.append(np.copy(base_grid))
        output_grids.append(np.copy(base_grid_mirrored))

        pair_count += 1

    if attempts == max_attempts:
        print("make_mirrored_obj_grids: Failed to create grids after maximum attempts")
        return None, None, None

    return flip_type, input_grids, output_grids


def make_scaled_obj_grids(
    min_grid_dim, max_grid_dim, num_train_tasks, num_test_tasks, max_attempts=1000
):
    input_grids = []
    output_grids = []

    min_sub_grid_dim = 2
    max_sub_grid_dim = 5

    rand_rot = randint(0, 3)
    rand_scale_or_shrink = randint(0, 100)

    attempts = 0
    pair_count = 0
    while pair_count < num_train_tasks + num_test_tasks and attempts < max_attempts:
        attempts += 1

        grid_x_dim = randint(min_grid_dim, max_grid_dim)
        grid_y_dim = randint(min_grid_dim, max_grid_dim)

        sub_grid_x_dim = randint(min_sub_grid_dim, max_sub_grid_dim)
        sub_grid_y_dim = randint(min_sub_grid_dim, max_sub_grid_dim)

        size = sub_grid_x_dim * sub_grid_y_dim
        permuted_array = np.random.randint(10, size=size)

        if check_zero_rows_cols(permuted_array, sub_grid_x_dim, sub_grid_y_dim):
            continue

        base_grid = np.zeros((grid_x_dim, grid_y_dim), dtype=int)
        base_grid_scaled = np.zeros((grid_x_dim, grid_y_dim), dtype=int)

        sub_grid = permuted_array.reshape((sub_grid_x_dim, sub_grid_y_dim))

        # Place sub-grid randomly inside of base_grid
        rand_start_x_index = randint(0, grid_x_dim - sub_grid_x_dim)
        rand_start_y_index = randint(0, grid_y_dim - sub_grid_y_dim)
        base_grid[
            rand_start_x_index : rand_start_x_index + sub_grid_x_dim,
            rand_start_y_index : rand_start_y_index + sub_grid_y_dim,
        ] = sub_grid

        input_grid = np.copy(base_grid)

        output_grid = np.zeros((grid_x_dim, grid_y_dim), dtype=int)
        scaled_sub_grid = np.copy(sub_grid)
        scaled_sub_grid_x_dim = 2 * sub_grid_x_dim
        scaled_sub_grid_y_dim = 2 * sub_grid_y_dim
        # Scaling up the original array
        k = 2
        scaled_sub_grid = np.kron(scaled_sub_grid, np.ones((k, k)))

        if (
            rand_start_x_index + scaled_sub_grid_x_dim > grid_x_dim - 1
            or rand_start_y_index + scaled_sub_grid_y_dim > grid_y_dim - 1
        ):
            continue
        else:
            output_grid[
                rand_start_x_index : rand_start_x_index + scaled_sub_grid_x_dim,
                rand_start_y_index : rand_start_y_index + scaled_sub_grid_y_dim,
            ] = scaled_sub_grid

            if rand_scale_or_shrink < 50:
                input_grid_2 = np.copy(output_grid)
                output_grid = np.copy(input_grid)
                input_grid = np.copy(input_grid_2)

            if rand_rot > 0 and rand_rot < 4:
                input_grid = np.rot90(input_grid, k=rand_rot)
                output_grid = np.rot90(output_grid, k=rand_rot)

            input_grids.append(input_grid)
            output_grids.append(output_grid)

            pair_count += 1

    if attempts == max_attempts:
        print("make_scaled_obj_grids: Failed to create grids after maximum attempts")
        return None, None, None

    return rand_rot, rand_scale_or_shrink, input_grids, output_grids


def make_swapped_color_grids(
    min_grid_dim, max_grid_dim, num_train_tasks, num_test_tasks, max_attempts=1000
):
    input_grids = []
    output_grids = []

    colors = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    permuted_colors = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    color_permuted_num = randint(1, 9)
    colors_used_in_train_tasks = []

    attempts = 0
    pair_count = 0
    while pair_count < num_train_tasks + num_test_tasks and attempts < max_attempts:
        attempts += 1

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
        for i in range(len(colors)):
            swap_index = i + color_permuted_num
            if swap_index >= len(colors):
                swap_index = (i + color_permuted_num) - len(colors)
            permuted_colors[i] = colors[swap_index]
        mapping = dict(zip(colors, permuted_colors))
        for x in range(grid_x_dim):
            for y in range(grid_y_dim):
                output_grid[x, y] = mapping[output_grid[x, y]]

        input_grids.append(input_grid)
        output_grids.append(output_grid)

        pair_count += 1

    if attempts == max_attempts:
        print("make_swapped_color_grids: Failed to create grids after maximum attempts")
        return None, None

    return input_grids, output_grids


def make_same_shape_grids(
    min_grid_dim, max_grid_dim, num_train_tasks, num_test_tasks, max_attempts=1000
):
    input_grids = []
    output_grids = []

    keep_same = randint(0, 1)

    rand_rot_or_flip = randint(0, 5)

    attempts = 0
    pair_count = 0
    while pair_count < num_train_tasks + num_test_tasks and attempts < max_attempts:
        attempts += 1

        grid_x_dim = randint(min_grid_dim, max_grid_dim)
        grid_y_dim = randint(min_grid_dim, max_grid_dim)

        #################### obj 1 ##################################################
        min_sub_grid_dim = 2
        max_sub_grid_dim = 3
        sub_grid_1_x_dim = randint(min_sub_grid_dim, max_sub_grid_dim)
        sub_grid_1_y_dim = randint(min_sub_grid_dim, max_sub_grid_dim)
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
        max_loop_count = 1000
        loop_count = 0
        while black_row_or_column or overlapping and loop_count < max_loop_count:
            loop_count += 1
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

        if loop_count >= max_loop_count:
            continue
        #################### obj 3 ##################################################
        black_row_or_column = True
        overlapping = True
        same_as_sub_grid_1 = True
        obj_3_loop_count = 0
        loop_count = 0
        while (
            black_row_or_column
            or overlapping
            or same_as_sub_grid_1
            and loop_count < max_loop_count
        ):
            loop_count += 1
            obj_3_loop_count += 1
            if obj_3_loop_count > 100:
                print("looping too much", obj_3_loop_count)
            black_row_or_column = False
            overlapping = False
            same_as_sub_grid_1 = False

            sub_grid_3_x_dim = randint(min_sub_grid_dim, max_sub_grid_dim)
            sub_grid_3_y_dim = randint(min_sub_grid_dim, max_sub_grid_dim)

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

        if loop_count >= max_loop_count:
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

    if attempts == max_attempts:
        print("make_same_shape_grids: Failed to create grids after maximum attempts")
        return None, None, None

    return keep_same, input_grids, output_grids


def make_fill_pattern_holes_grids(
    min_grid_dim, max_grid_dim, num_train_tasks, num_test_tasks, max_attempts=1000
):
    input_grids = []
    output_grids = []

    attempts = 0
    pair_count = 0
    while pair_count < num_train_tasks + num_test_tasks and attempts < max_attempts:
        attempts += 1

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
        while no_black_squares:
            for x in range(grid_x_dim):
                for y in range(grid_y_dim):
                    make_black = randint(1, 100)
                    if make_black > 95:
                        input_grid[x][y] = 0
                        no_black_squares = False
        input_grids.append(input_grid)

        pair_count += 1

    if attempts == max_attempts:
        print(
            "make_fill_pattern_holes_grids: Failed to create grids after maximum attempts"
        )
        return None, None

    return input_grids, output_grids


def make_fill_rotated_pattern_holes_grids(
    min_grid_dim, max_grid_dim, num_train_tasks, num_test_tasks, max_attempts=1000
):
    input_grids = []
    output_grids = []

    attempts = 0
    pair_count = 0
    while pair_count < num_train_tasks + num_test_tasks and attempts < max_attempts:
        attempts += 1

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

    if attempts == max_attempts:
        print(
            "make_fill_rotated_pattern_holes_grids: Failed to create grids after maximum attempts"
        )
        return None, None

    return input_grids, output_grids


def make_fill_surrounded_grids(
    min_grid_dim, max_grid_dim, num_train_tasks, num_test_tasks, max_attempts=1000
):
    base_grids = []
    input_grids = []
    output_grids = []

    interior_color = randint(1, 9)
    surround_color = randint(1, 9)
    while surround_color == interior_color:
        surround_color = randint(1, 9)

    fill_exterior = randint(0, 99)

    attempts = 0
    pair_count = 0
    while pair_count < num_train_tasks + num_test_tasks and attempts < max_attempts:
        attempts += 1

        grid_x_dim = randint(min_grid_dim, max_grid_dim)
        grid_y_dim = randint(min_grid_dim, max_grid_dim)

        sub_grid_min_dim = 1
        sub_grid_max_dim = min_grid_dim - 2
        sub_grid_x_dim = randint(sub_grid_min_dim, sub_grid_max_dim)
        sub_grid_y_dim = randint(sub_grid_min_dim, sub_grid_max_dim)

        size = sub_grid_x_dim * sub_grid_y_dim
        permuted_array = np.random.randint(10, size=size)

        if check_zero_rows_cols(permuted_array, sub_grid_x_dim, sub_grid_y_dim):
            # print("zero row or column: task_count " + str(task_count))
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

    if attempts == max_attempts:
        print(
            "make_fill_surrounded_grids: Failed to create grids after maximum attempts"
        )
        return None, None

    return fill_exterior, input_grids, output_grids


def make_gravity_grids(
    min_grid_dim, max_grid_dim, num_train_tasks, num_test_tasks, max_attempts=1000
):
    base_grids = []
    input_grids = []
    output_grids = []

    rot_num = randint(0, 3)

    attempts = 0
    pair_count = 0
    while pair_count < num_train_tasks + num_test_tasks and attempts < max_attempts:
        attempts += 1

        grid_x_dim = randint(min_grid_dim, max_grid_dim)
        grid_y_dim = randint(min_grid_dim, max_grid_dim)

        loop_again = False
        min_sub_grid_dim = 2
        max_sub_grid_x_dim = grid_x_dim - 1
        max_sub_grid_y_dim = grid_y_dim - 1
        sub_grid_x_dim = randint(min_sub_grid_dim, max_sub_grid_x_dim)
        sub_grid_y_dim = randint(min_sub_grid_dim, max_sub_grid_y_dim)
        size = sub_grid_x_dim * sub_grid_y_dim
        permuted_array = np.random.randint(10, size=size)

        if check_zero_rows_cols(permuted_array, sub_grid_x_dim, sub_grid_y_dim):
            # print("zero row or column: task_count " + str(task_count))
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

    if attempts == max_attempts:
        print("make_gravity_grids: Failed to create grids after maximum attempts")
        return None, None

    return rot_num, input_grids, output_grids


def make_isolate_obj_grids(
    min_grid_dim, max_grid_dim, num_train_tasks, num_test_tasks, max_attempts=1000
):
    input_grids = []
    output_grids = []

    min_sub_grid_dim = 2
    max_sub_grid_dim = 7
    sub_grid_x_dim = randint(min_sub_grid_dim, max_sub_grid_dim)
    sub_grid_y_dim = randint(min_sub_grid_dim, max_sub_grid_dim)
    size = sub_grid_x_dim * sub_grid_y_dim
    permuted_array = np.random.randint(10, size=size)

    while check_zero_rows_cols(permuted_array, sub_grid_x_dim, sub_grid_y_dim):
        sub_grid_x_dim = randint(min_sub_grid_dim, max_sub_grid_dim)
        sub_grid_y_dim = randint(min_sub_grid_dim, max_sub_grid_dim)
        size = sub_grid_x_dim * sub_grid_y_dim
        permuted_array = np.random.randint(10, size=size)

    sub_grid = permuted_array.reshape((sub_grid_x_dim, sub_grid_y_dim))

    input_grids = []
    output_grids = []
    flattened_isolated_obj_grids = set()

    attempts = 0
    pair_count = 0
    while pair_count < num_train_tasks + num_test_tasks and attempts < max_attempts:
        attempts += 1

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
            flattened_isolated_obj_grids.add(flat_tuple)

            output_grid = np.copy(isolated_obj_grid)
            output_grids.append(output_grid)

            make_input_grid = True
            while make_input_grid:
                make_input_grid = False
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

            pair_count += 1

    if attempts == max_attempts:
        print("make_isolate_obj_grids: Failed to create grids after maximum attempts")
        return None, None

    return input_grids, output_grids


def make_largest_smallest_obj_grids(
    min_grid_dim, max_grid_dim, num_train_tasks, num_test_tasks, max_attempts=1000
):
    input_grids = []
    output_grids = []

    keep_larger = randint(0, 1)
    rand_rot_or_flip = randint(0, 5)

    attempts = 0
    pair_count = 0
    while pair_count < num_train_tasks + num_test_tasks and attempts < max_attempts:
        attempts += 1

        grid_x_dim = randint(min_grid_dim, max_grid_dim)
        grid_y_dim = randint(min_grid_dim, max_grid_dim)

        #################### obj 1 ##################################################
        min_sub_grid_dim = 2
        max_sub_grid_dim = 4
        sub_grid_1_x_dim = randint(min_sub_grid_dim, max_sub_grid_dim)
        sub_grid_1_y_dim = randint(min_sub_grid_dim, max_sub_grid_dim)

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
        while black_row_or_column or same_square_count or overlapping:
            black_row_or_column = False
            same_square_count = False
            overlapping = False

            sub_grid_2_x_dim = randint(min_sub_grid_dim, max_sub_grid_dim)
            sub_grid_2_y_dim = randint(min_sub_grid_dim, max_sub_grid_dim)

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

    if attempts == max_attempts:
        print(
            "make_largest_smallest_obj_grids: Failed to create grids after maximum attempts"
        )
        return None, None

    return keep_larger, input_grids, output_grids


def make_mask_obj_data_grids(
    min_grid_dim, max_grid_dim, num_train_tasks, num_test_tasks, max_attempts=1000
):
    input_grids = []
    output_grids = []

    min_sub_grid_dim = 2
    max_sub_grid_dim = 4

    sub_grid_x_dim = randint(min_sub_grid_dim, max_sub_grid_dim)
    sub_grid_y_dim = randint(min_sub_grid_dim, max_sub_grid_dim)
    permuted_array = np.random.randint(10, size=sub_grid_x_dim * sub_grid_y_dim)

    while check_zero_rows_cols(permuted_array, sub_grid_x_dim, sub_grid_y_dim):
        sub_grid_x_dim = randint(min_sub_grid_dim, max_sub_grid_dim)
        sub_grid_y_dim = randint(min_sub_grid_dim, max_sub_grid_dim)
        permuted_array = np.random.randint(10, size=sub_grid_x_dim * sub_grid_y_dim)

    sub_grid = permuted_array.reshape((sub_grid_x_dim, sub_grid_y_dim))

    input_grids = []
    isolated_obj_grids = []
    output_grids = []
    flattened_isolated_obj_grids = set()
    mask_color = randint(0, 9)

    attempts = 0
    pair_count = 0
    while pair_count < num_train_tasks + num_test_tasks and attempts < max_attempts:
        attempts += 1

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

            make_input_grid = True
            while make_input_grid:
                make_input_grid = False
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

                    pair_count += 1

    if attempts == max_attempts:
        print("make_mask_obj_data_grids: Failed to create grids after maximum attempts")
        return None, None

    return mask_color, input_grids, output_grids


def make_mirror_grids(
    min_grid_dim, max_grid_dim, num_train_tasks, num_test_tasks, max_attempts=1000
):
    input_grids = []
    output_grids = []

    flip_type = randint(0, 1)

    attempts = 0
    pair_count = 0
    while pair_count < num_train_tasks + num_test_tasks and attempts < max_attempts:
        attempts += 1

        grid_x_dim = randint(min_grid_dim, max_grid_dim)
        grid_y_dim = randint(min_grid_dim, max_grid_dim)

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

        pair_count += 1

    if attempts == max_attempts:
        print("make_mirror_grids: Failed to create grids after maximum attempts")
        return None, None

    return flip_type, input_grids, output_grids


def make_most_freq_in_row_grids(
    min_grid_dim, max_grid_dim, num_train_tasks, num_test_tasks, max_attempts=1000
):
    input_grids = []
    output_grids = []

    find_column = randint(0, 1)

    find_min_color = 0  # randint(0, 1)

    attempts = 0
    pair_count = 0
    while pair_count < num_train_tasks + num_test_tasks and attempts < max_attempts:
        attempts += 1

        do_again = True
        while do_again:
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
        if find_column > 0:
            input_grid = np.rot90(input_grid, k=1)
            output_grid = np.rot90(output_grid, k=1)

        input_grids.append(input_grid)
        output_grids.append(output_grid)

        pair_count += 1

    if attempts == max_attempts:
        print(
            "make_most_freq_in_row_grids: Failed to create grids after maximum attempts"
        )
        return None, None

    return find_column, input_grids, output_grids


def make_multiple_objs_grids(
    min_grid_dim, max_grid_dim, num_train_tasks, num_test_tasks, max_attempts=10000
):
    input_grids = []
    output_grids = []

    attempts = 0
    pair_count = 0
    loop_through_whole_thing = False
    while pair_count < num_train_tasks + num_test_tasks and attempts < max_attempts:
        attempts += 1

        grid_x_dim = randint(min_grid_dim, max_grid_dim)
        grid_y_dim = randint(min_grid_dim, max_grid_dim)

        min_sub_grid_dim = 2
        max_sub_grid_x_dim = (grid_x_dim // 2) - 1
        max_sub_grid_y_dim = (grid_y_dim // 2) - 1

        isolated_obj_grid = np.zeros((grid_x_dim, grid_y_dim), dtype=int)

        sub_grid_x_dim = randint(min_sub_grid_dim, max_sub_grid_x_dim)
        sub_grid_y_dim = randint(min_sub_grid_dim, max_sub_grid_y_dim)
        size = sub_grid_x_dim * sub_grid_y_dim
        permuted_array = np.random.randint(10, size=size)

        while check_zero_rows_cols(permuted_array, sub_grid_x_dim, sub_grid_y_dim):
            sub_grid_x_dim = randint(min_sub_grid_dim, max_sub_grid_x_dim)
            sub_grid_y_dim = randint(min_sub_grid_dim, max_sub_grid_y_dim)
            size = sub_grid_x_dim * sub_grid_y_dim
            permuted_array = np.random.randint(10, size=size)

        sub_grid = permuted_array.reshape((sub_grid_x_dim, sub_grid_y_dim))

        num_objs = randint(2, 3)
        if (sub_grid_x_dim == 4 and sub_grid_y_dim == 3) or (
            sub_grid_x_dim == 3 and sub_grid_y_dim == 4
        ):
            num_objs = 2

        i = 0
        while i < num_objs:
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
                while try_again and try_again_count < 100:
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
                if try_again_count >= 100:
                    loop_through_whole_thing = True
                    break

                for x in range(grid_x_dim):
                    for y in range(grid_y_dim):
                        if candidate_isolated_obj_grid[x][y] > 0:
                            isolated_obj_grid[x][y] = candidate_isolated_obj_grid[x][y]
                i += 1

        if loop_through_whole_thing:
            continue
        else:
            output_grid = np.copy(isolated_obj_grid)
            output_grids.append(output_grid)
            input_grid = np.copy(output_grid)
            for x in range(grid_x_dim):
                for y in range(grid_y_dim):
                    if input_grid[x][y] == 0:
                        input_grid[x][y] = randint(0, 9)
            input_grids.append(input_grid)
            pair_count += 1

    if attempts == max_attempts:
        print("make_multiple_objs_grids: Failed to create grids after maximum attempts")
        input_grids, output_grids = make_multiple_objs_grids(
            min_grid_dim, max_grid_dim, num_train_tasks, num_test_tasks
        )

    return input_grids, output_grids


def make_rays_grids(
    min_grid_dim, max_grid_dim, num_train_tasks, num_test_tasks, max_attempts=1000
):
    input_grids = []
    output_grids = []

    rot_num = randint(0, 3)

    attempts = 0
    pair_count = 0
    while pair_count < num_train_tasks + num_test_tasks and attempts < max_attempts:
        attempts += 1

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

    if attempts == max_attempts:
        print("make_rays_grids: Failed to create grids after maximum attempts")
        return None, None

    return input_grids, output_grids


def make_rotate_grids(
    min_grid_dim, max_grid_dim, num_train_tasks, num_test_tasks, max_attempts=1000
):
    input_grids = []
    output_grids = []

    rot_num = randint(1, 3)

    attempts = 0
    pair_count = 0
    while pair_count < num_train_tasks + num_test_tasks and attempts < max_attempts:
        attempts += 1

        grid_x_dim = randint(min_grid_dim, max_grid_dim)
        grid_y_dim = randint(min_grid_dim, max_grid_dim)

        input_grid = np.zeros((grid_x_dim, grid_y_dim), dtype=int)
        for x in range(grid_x_dim):
            for y in range(grid_y_dim):
                if input_grid[x][y] == 0:
                    input_grid[x][y] = randint(0, 9)
        input_grids.append(input_grid)

        output_grid = np.copy(input_grid)
        output_grid = np.rot90(output_grid, k=rot_num)
        output_grids.append(output_grid)

        pair_count += 1

    if attempts == max_attempts:
        print("make_rotate_grids: Failed to create grids after maximum attempts")
        return None, None

    return rot_num, input_grids, output_grids


def make_same_color_objs_grids(
    min_grid_dim, max_grid_dim, num_train_tasks, num_test_tasks, max_attempts=1000
):
    input_grids = []
    output_grids = []

    keep_same = randint(0, 1)

    rand_rot_or_flip = randint(0, 5)

    attempts = 0
    pair_count = 0
    while pair_count < num_train_tasks + num_test_tasks and attempts < max_attempts:
        attempts += 1

        grid_x_dim = randint(min_grid_dim, max_grid_dim)
        grid_y_dim = randint(min_grid_dim, max_grid_dim)

        #################### obj 1 ##################################################
        min_sub_grid_dim = 2
        max_sub_grid_dim = 2
        sub_grid_1_x_dim = randint(min_sub_grid_dim, max_sub_grid_dim)
        sub_grid_1_y_dim = randint(min_sub_grid_dim, max_sub_grid_dim)
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
        while black_row_or_column or same_square_count or overlapping:
            black_row_or_column = False
            same_square_count = False
            overlapping = False

            sub_grid_2_x_dim = randint(min_sub_grid_dim, max_sub_grid_dim)
            sub_grid_2_y_dim = randint(min_sub_grid_dim, max_sub_grid_dim)

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
        while black_row_or_column or same_square_count or overlapping:
            obj_3_loop_count += 1
            black_row_or_column = False
            same_square_count = False
            overlapping = False

            sub_grid_3_x_dim = randint(min_sub_grid_dim, max_sub_grid_dim)
            sub_grid_3_y_dim = randint(min_sub_grid_dim, max_sub_grid_dim)

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

    if attempts == max_attempts:
        print("make_rotate_grids: Failed to create grids after maximum attempts")
        return None, None

    return keep_same, input_grids, output_grids


###############################################################################
def create_instruction(train_input_grids, train_output_grids, test_input_grids):
    instruction = "Please solve this abstract reasoning puzzle: "
    instruction += (
        "Below are input/output pairs of grids in two dimensional array format. "
    )
    instruction += "Each grid square contains an integer from 0 to 9. "
    instruction += "These intergers are symbolic representations of colors, and should not be thought of as scalar values. "
    instruction += "The integer 0 represents an empty grid square. The other integers 1 through 9 represent "
    instruction += "colored tiles that occupy the grid square positions corresponding to their locations in the "
    instruction += "two dimensional array. "
    instruction += (
        "Given the following "
        + str(len(train_input_grids))
        + " input/output train pairs: "
    )

    for i, (train_input_grid, train_output_grid) in enumerate(
        zip(train_input_grids, train_output_grids)
    ):
        instruction += f"Train_Input_{i+1}=["
        for j in range(len(train_input_grid)):
            instruction += "["
            for k in range(len(train_input_grid[j])):
                instruction += str(train_input_grid[j][k])
                if k != len(train_input_grid[j]) - 1:
                    instruction += ","
            instruction += "]"
            if j != len(train_input_grid) - 1:
                instruction += ","
        instruction += "]"
        instruction += f" and Train_Output_{i+1}=["
        for j in range(len(train_output_grid)):
            instruction += "["
            for k in range(len(train_output_grid[j])):
                instruction += str(train_output_grid[j][k])
                if k != len(train_output_grid[j]) - 1:
                    instruction += ","
            instruction += "]"
            if j != len(train_output_grid) - 1:
                instruction += ","
        instruction += "]"
        if i != len(train_output_grids) - 1:
            instruction += ", "

    instruction += (
        " find the transformation from each input grid to output grid that is common to all "
        + str(len(train_input_grids))
        + " train pairs. "
    )
    instruction += "For example, tiles can be created, destroyed, moved, scaled, rotated, or change color. "
    instruction += "Groups of tiles that are adjacent to each other can be considered as objects, or sub-grids. "
    instruction += "Sub-grids can also be created, destroyed, moved, scaled, rotated, or change color. "
    instruction += " Grid tiles and sub-grids can spawn into lines that extend vertically, horizontally, or diagonally. "
    instruction += (
        "Devise a candidate transformation and test it on each of the train grids. "
    )
    instruction += "If the candidate transformation fails to produce the correct train output grid for any of the train pairs, "

    instruction += "then that candidate transformation is incorrect. Devise a new candidate transformation and re-test on the train pairs. "
    instruction += "Once you have discovered the correct common tranformation that works on all train pairs, "
    instruction += (
        "apply it to the following test input grid to get the test output grid: "
    )

    for i, (test_input_grid) in enumerate(test_input_grids):
        instruction += f"Test_Input_{i+1}=["
        for j in range(len(test_input_grid)):
            instruction += "["
            for k in range(len(test_input_grid[j])):
                instruction += str(test_input_grid[j][k])
                if k != len(test_input_grid[j]) - 1:
                    instruction += ","
            instruction += "]"
            if j != len(test_input_grid) - 1:
                instruction += ","
        instruction += "]"
    instruction += "."
    instruction += " Use your knowledge of core principals in physics, mathematics, chain of thought reasoning to solve this puzzle. "

    return instruction


def create_rotate_obj_puzzle_prompt(
    min_grid_dim, max_grid_dim, num_train_tasks, num_test_tasks, tokenizer
):
    (
        rand_rot_grid,
        rand_flip_grid,
        rot_num,
        input_grids,
        output_grids,
    ) = make_rotate_obj_grids(
        min_grid_dim, max_grid_dim, num_train_tasks, num_test_tasks
    )

    train_input_grids = input_grids[:num_train_tasks]
    train_output_grids = output_grids[:num_train_tasks]

    test_input_grids = input_grids[num_train_tasks:]
    test_output_grids = output_grids[num_train_tasks:]

    instruction = create_instruction(
        train_input_grids, train_output_grids, test_input_grids
    )

    output = (
        "The candidate transformation is that the non-zero element sub-grid in each train input grid is rotated "
        + str(rot_num * 90)
    )

    if rand_rot_grid == 0 and rand_flip_grid == 0:
        # No grid transformation = rotate counter-clockwise around top-left corner
        output += " degrees counter-clockwise about its top left element to get the corresponding train output grid.  "
    elif rand_rot_grid == 0 and rand_flip_grid == 1:
        # Flip grid left-right, no grid rotation = rotate clockwise around top-right corner
        output += " degrees clockwise about its top right element to get the corresponding train output grid. "
    elif rand_rot_grid == 1 and rand_flip_grid == 0:
        # Rotate grid 90 degrees counter-clockwise, no flip = rotate counter-clockwise around bottom-left corner
        output += " degrees counter-clockwise about its bottom left element to get the corresponding train output grid. "
    elif rand_rot_grid == 1 and rand_flip_grid == 1:
        # Rotate grid 90 degrees counter-clockwise and flip left-right = rotate clockwise around bottom-right corner
        output += " degrees clockwise about its bottom right element to get the corresponding train output grid. "
    elif rand_rot_grid == 2 and rand_flip_grid == 0:
        # Rotate grid 180 degrees counter-clockwise = rotate counter-clockwise around bottom-right corner
        output += " degrees counter-clockwise about its bottom right element to get the corresponding train output grid. "
    elif rand_rot_grid == 2 and rand_flip_grid == 1:
        # Rotate grid 180 degrees counter-clockwise and flip left-right = rotate clockwise around bottom-left corner
        output += " degrees clockwise about its bottom left element to get the corresponding train output grid. "
    elif rand_rot_grid == 3 and rand_flip_grid == 0:
        # Rotate grid 270 degrees counter-clockwise = rotate counter-clockwise around top-right corner
        output += " degrees counter-clockwise about its top right element to get the corresponding train output grid. "
    else:
        # Rotate grid 270 degrees counter-clockwise and flip left-right = rotate clockwise around top-left corner
        output += " degrees clockwise about its top left element to get the corresponding train output grid. "

    output += "I've tested this candidate transformation on each of the train pairs, and it works for all of them. "
    output += "Therefore, I've applied the candidate transformation to the test input grid to get the following test output grid: "

    for i, (test_output_grid) in enumerate(test_output_grids):
        output += f"Test_Output_{i+1}=["
        for j in range(len(test_output_grid)):
            output += "["
            for k in range(len(test_output_grid[j])):
                output += str(test_output_grid[j][k])
                if k != len(test_output_grid[j]) - 1:
                    output += ","
            output += "]"
            if j != len(test_output_grid) - 1:
                output += ","
        output += "]"
        if i != len(test_output_grids) - 1:
            output += ", "
    output += "."

    # Tokenize the request text
    prompt = instruction + " " + output
    tokenized_request = tokenizer.tokenize(prompt)

    # Get the token length
    token_length = len(tokenized_request)

    return instruction, output, token_length


def create_move_obj_puzzle_prompt(
    min_grid_dim, max_grid_dim, num_train_tasks, num_test_tasks, tokenizer
):
    row_move, column_move, input_grids, output_grids = make_move_obj_grids(
        min_grid_dim, max_grid_dim, num_train_tasks, num_test_tasks
    )

    train_input_grids = input_grids[:num_train_tasks]
    train_output_grids = output_grids[:num_train_tasks]

    test_input_grids = input_grids[num_train_tasks:]
    test_output_grids = output_grids[num_train_tasks:]

    instruction = create_instruction(
        train_input_grids, train_output_grids, test_input_grids
    )

    output = (
        "The candidate transformation is that the non-zero element sub-grid in each train input grid is moved "
        + str(row_move)
        + " units vertically and "
        + str(column_move)
        + " units horizontally to get the corresponding train output grid. "
    )
    output += "I've tested this candidate transformation on each of the train pairs, and it works for all of them. "
    output += "Therefore, I've applied the candidate transformation to the test input grid to get the following test output grid: "
    for i, (test_output_grid) in enumerate(test_output_grids):
        output += f"Test_Output_{i+1}=["
        for j in range(len(test_output_grid)):
            output += "["
            for k in range(len(test_output_grid[j])):
                output += str(test_output_grid[j][k])
                if k != len(test_output_grid[j]) - 1:
                    output += ","
            output += "]"
            if j != len(test_output_grid) - 1:
                output += ","
        output += "]"
        if i != len(test_output_grids) - 1:
            output += ", "
    output += "."

    # Tokenize the request text
    prompt = instruction + " " + output
    tokenized_request = tokenizer.tokenize(prompt)

    # Get the token length
    token_length = len(tokenized_request)

    return instruction, output, token_length


def create_mirrored_obj_puzzle_prompt(
    min_grid_dim, max_grid_dim, num_train_tasks, num_test_tasks, tokenizer
):
    flip_type, input_grids, output_grids = make_mirrored_obj_grids(
        min_grid_dim, max_grid_dim, num_train_tasks, num_test_tasks
    )

    train_input_grids = input_grids[:num_train_tasks]
    train_output_grids = output_grids[:num_train_tasks]

    test_input_grids = input_grids[num_train_tasks:]
    test_output_grids = output_grids[num_train_tasks:]

    instruction = create_instruction(
        train_input_grids, train_output_grids, test_input_grids
    )

    output = "The candidate transformation is that the non-zero element sub-grid in each train input grid is mirrored "
    if flip_type == 0:
        output += "vertically"
    else:
        output += "horizontally"
    output += " about its center to get the corresponding train output grid. "
    output += "I've tested this candidate transformation on each of the train pairs, and it works for all of them. "
    output += "Therefore, I've applied the candidate transformation to the test input grid to get the following test output grid: "
    for i, (test_output_grid) in enumerate(test_output_grids):
        output += f"Test_Output_{i+1}=["
        for j in range(len(test_output_grid)):
            output += "["
            for k in range(len(test_output_grid[j])):
                output += str(test_output_grid[j][k])
                if k != len(test_output_grid[j]) - 1:
                    output += ","
            output += "]"
            if j != len(test_output_grid) - 1:
                output += ","
        output += "]"
        if i != len(test_output_grids) - 1:
            output += ", "
    output += "."

    # Tokenize the request text
    prompt = instruction + " " + output
    tokenized_request = tokenizer.tokenize(prompt)

    # Get the token length
    token_length = len(tokenized_request)

    return instruction, output, token_length


def create_scaled_obj_puzzle_prompt(
    min_grid_dim, max_grid_dim, num_train_tasks, num_test_tasks, tokenizer
):
    rand_rot, rand_scale_or_shrink, input_grids, output_grids = make_scaled_obj_grids(
        min_grid_dim, max_grid_dim, num_train_tasks, num_test_tasks
    )

    train_input_grids = input_grids[:num_train_tasks]
    train_output_grids = output_grids[:num_train_tasks]

    test_input_grids = input_grids[num_train_tasks:]
    test_output_grids = output_grids[num_train_tasks:]

    instruction = create_instruction(
        train_input_grids, train_output_grids, test_input_grids
    )

    output = "The candidate transformation is that the non-zero element sub-grid in each train input grid is scaled "
    if rand_scale_or_shrink > 50:
        output += "up"
    else:
        output += "down"
    output += " by 2 about its "
    if rand_rot == 0:
        output += "top left element to get the corresponding train output grid. "
    elif rand_rot == 1:
        output += "bottom left element to get the corresponding train output grid. "
    elif rand_rot == 2:
        output += "bottom right element to get the corresponding train output grid. "
    else:
        output += "top right element to get the corresponding train output grid. "
    output += "I've tested this candidate transformation on each of the train pairs, and it works for all of them. "
    output += "Therefore, I've applied the candidate transformation to the test input grid to get the following test output grid: "

    for i, (test_output_grid) in enumerate(test_output_grids):
        output += f"Test_Output_{i+1}=["
        for j in range(len(test_output_grid)):
            output += "["
            for k in range(len(test_output_grid[j])):
                output += str(test_output_grid[j][k])
                if k != len(test_output_grid[j]) - 1:
                    output += ","
            output += "]"
            if j != len(test_output_grid) - 1:
                output += ","
        output += "]"
        if i != len(test_output_grids) - 1:
            output += ", "
    output += "."

    # Tokenize the request text
    prompt = instruction + " " + output
    tokenized_request = tokenizer.tokenize(prompt)

    # Get the token length
    token_length = len(tokenized_request)

    return instruction, output, token_length


def create_swapped_color_grids_prompt(
    min_grid_dim, max_grid_dim, num_train_tasks, num_test_tasks, tokenizer
):
    input_grids, output_grids = make_swapped_color_grids(
        min_grid_dim, max_grid_dim, num_train_tasks, num_test_tasks
    )

    train_input_grids = input_grids[:num_train_tasks]
    train_output_grids = output_grids[:num_train_tasks]

    test_input_grids = input_grids[num_train_tasks:]
    test_output_grids = output_grids[num_train_tasks:]

    instruction = create_instruction(
        train_input_grids, train_output_grids, test_input_grids
    )

    output = "The candidate transformation is that certain integers in each train input grid are "
    output += "swapped with other integers to get the corresponding train output grid. "
    output += "These swapped integers are the same throughout all of the train sets. "
    output += "I've tested this candidate transformation on each of the train pairs, and it works for all of them. "
    output += "Therefore, I've applied the candidate transformation to the test input grid to get the following test output grid: "

    for i, (test_output_grid) in enumerate(test_output_grids):
        output += f"Test_Output_{i+1}=["
        for j in range(len(test_output_grid)):
            output += "["
            for k in range(len(test_output_grid[j])):
                output += str(test_output_grid[j][k])
                if k != len(test_output_grid[j]) - 1:
                    output += ","
            output += "]"
            if j != len(test_output_grid) - 1:
                output += ","
        output += "]"
        if i != len(test_output_grids) - 1:
            output += ", "
    output += "."

    # Tokenize the request text
    prompt = instruction + " " + output
    tokenized_request = tokenizer.tokenize(prompt)

    # Get the token length
    token_length = len(tokenized_request)

    return instruction, output, token_length


def create_same_shape_grids_prompt(
    min_grid_dim, max_grid_dim, num_train_tasks, num_test_tasks, tokenizer
):
    keep_same, input_grids, output_grids = make_same_shape_grids(
        min_grid_dim, max_grid_dim, num_train_tasks, num_test_tasks
    )

    train_input_grids = input_grids[:num_train_tasks]
    train_output_grids = output_grids[:num_train_tasks]

    test_input_grids = input_grids[num_train_tasks:]
    test_output_grids = output_grids[num_train_tasks:]

    instruction = create_instruction(
        train_input_grids, train_output_grids, test_input_grids
    )

    output = "The candidate transformation is that sub-grids in each train input grid that are "
    if keep_same > 0:
        output += "not the same shape are removed to get the corresponding train output grid. "
    else:
        output += (
            "the same shape are removed to get the corresponding train output grid. "
        )
    output += "I've tested this candidate transformation on each of the train pairs, and it works for all of them. "
    output += "Therefore, I've applied the candidate transformation to the test input grid to get the following test output grid: "

    for i, (test_output_grid) in enumerate(test_output_grids):
        output += f"Test_Output_{i+1}=["
        for j in range(len(test_output_grid)):
            output += "["
            for k in range(len(test_output_grid[j])):
                output += str(test_output_grid[j][k])
                if k != len(test_output_grid[j]) - 1:
                    output += ","
            output += "]"
            if j != len(test_output_grid) - 1:
                output += ","
        output += "]"
        if i != len(test_output_grids) - 1:
            output += ", "
    output += "."

    # Tokenize the request text
    prompt = instruction + " " + output
    tokenized_request = tokenizer.tokenize(prompt)

    # Get the token length
    token_length = len(tokenized_request)

    return instruction, output, token_length


def create_fill_pattern_holes_grids_prompt(
    min_grid_dim, max_grid_dim, num_train_tasks, num_test_tasks, tokenizer
):
    input_grids, output_grids = make_fill_pattern_holes_grids(
        min_grid_dim, max_grid_dim, num_train_tasks, num_test_tasks
    )

    train_input_grids = input_grids[:num_train_tasks]
    train_output_grids = output_grids[:num_train_tasks]

    test_input_grids = input_grids[num_train_tasks:]
    test_output_grids = output_grids[num_train_tasks:]

    instruction = create_instruction(
        train_input_grids, train_output_grids, test_input_grids
    )

    output = "There are patterns in each quadrant of each train input grid that are repeated and flipped depending on their quadrant position in the grid. "
    output += (
        "There are also holes (grid squares with integer 0) in the train input grids. "
    )
    output += (
        "The candidate transformation is that the holes in each train input grid are "
    )
    output += "filled in with the corresponding pattern to get the corresponding train output grid. "
    output += "I've tested this candidate transformation on each of the train pairs, and it works for all of them. "
    output += "Therefore, I've applied the candidate transformation to the test input grid to get the following test output grid: "

    for i, (test_output_grid) in enumerate(test_output_grids):
        output += f"Test_Output_{i+1}=["
        for j in range(len(test_output_grid)):
            output += "["
            for k in range(len(test_output_grid[j])):
                output += str(test_output_grid[j][k])
                if k != len(test_output_grid[j]) - 1:
                    output += ","
            output += "]"
            if j != len(test_output_grid) - 1:
                output += ","
        output += "]"
        if i != len(test_output_grids) - 1:
            output += ", "
    output += "."

    # Tokenize the request text
    prompt = instruction + " " + output
    tokenized_request = tokenizer.tokenize(prompt)

    # Get the token length
    token_length = len(tokenized_request)

    return instruction, output, token_length


def create_fill_rotated_pattern_holes_grids_prompt(
    min_grid_dim, max_grid_dim, num_train_tasks, num_test_tasks, tokenizer
):
    input_grids, output_grids = make_fill_rotated_pattern_holes_grids(
        min_grid_dim, max_grid_dim, num_train_tasks, num_test_tasks
    )

    train_input_grids = input_grids[:num_train_tasks]
    train_output_grids = output_grids[:num_train_tasks]

    test_input_grids = input_grids[num_train_tasks:]
    test_output_grids = output_grids[num_train_tasks:]

    instruction = create_instruction(
        train_input_grids, train_output_grids, test_input_grids
    )

    output = "There are patterns in each quadrant of each train input grid that are repeated and rotated depending on their quadrant position in the grid. "
    output += (
        "There are also holes (grid squares with integer 0) in the train input grids. "
    )
    output += (
        "The candidate transformation is that the holes in each train input grid are "
    )
    output += "filled in with the corresponding pattern to get the corresponding train output grid. Therefore, "
    output += "I've tested this candidate transformation on each of the train pairs, and it works for all of them. "
    output += "Therefore, I've applied the candidate transformation to the test input grid to get the following test output grid: "

    for i, (test_output_grid) in enumerate(test_output_grids):
        output += f"Test_Output_{i+1}=["
        for j in range(len(test_output_grid)):
            output += "["
            for k in range(len(test_output_grid[j])):
                output += str(test_output_grid[j][k])
                if k != len(test_output_grid[j]) - 1:
                    output += ","
            output += "]"
            if j != len(test_output_grid) - 1:
                output += ","
        output += "]"
        if i != len(test_output_grids) - 1:
            output += ", "
    output += "."

    # Tokenize the request text
    prompt = instruction + " " + output
    tokenized_request = tokenizer.tokenize(prompt)

    # Get the token length
    token_length = len(tokenized_request)

    return instruction, output, token_length


def create_fill_surrounded_grids_prompt(
    min_grid_dim, max_grid_dim, num_train_tasks, num_test_tasks, tokenizer
):
    fill_exterior, input_grids, output_grids = make_fill_surrounded_grids(
        min_grid_dim, max_grid_dim, num_train_tasks, num_test_tasks
    )

    train_input_grids = input_grids[:num_train_tasks]
    train_output_grids = output_grids[:num_train_tasks]

    test_input_grids = input_grids[num_train_tasks:]
    test_output_grids = output_grids[num_train_tasks:]

    instruction = create_instruction(
        train_input_grids, train_output_grids, test_input_grids
    )

    output = "There are empty grid squares (integer 0) in each train input grid that are surrounded by colored tiles. "
    if fill_exterior > 49:
        output += "The candidate transformation is that the empty grid squares that are not surrounded are filled with a certain color of tile that is common in all train output grids."
    else:
        output += "The candidate transformation is that the surrounded empty grid squares are filled with a certain color of tile that is common in all train output grids."
    output += "I've tested this candidate transformation on each of the train pairs, and it works for all of them. "
    output += "Therefore, I've applied the candidate transformation to the test input grid to get the following test output grid: "

    for i, (test_output_grid) in enumerate(test_output_grids):
        output += f"Test_Output_{i+1}=["
        for j in range(len(test_output_grid)):
            output += "["
            for k in range(len(test_output_grid[j])):
                output += str(test_output_grid[j][k])
                if k != len(test_output_grid[j]) - 1:
                    output += ","
            output += "]"
            if j != len(test_output_grid) - 1:
                output += ","
        output += "]"
        if i != len(test_output_grids) - 1:
            output += ", "
    output += "."

    # Tokenize the request text
    prompt = instruction + " " + output
    tokenized_request = tokenizer.tokenize(prompt)

    # Get the token length
    token_length = len(tokenized_request)

    return instruction, output, token_length


def create_gravity_grids_prompt(
    min_grid_dim, max_grid_dim, num_train_tasks, num_test_tasks, tokenizer
):
    rot_num, input_grids, output_grids = make_gravity_grids(
        min_grid_dim, max_grid_dim, num_train_tasks, num_test_tasks
    )

    train_input_grids = input_grids[:num_train_tasks]
    train_output_grids = output_grids[:num_train_tasks]

    test_input_grids = input_grids[num_train_tasks:]
    test_output_grids = output_grids[num_train_tasks:]

    instruction = create_instruction(
        train_input_grids, train_output_grids, test_input_grids
    )

    output = "The candidate transformation is to conceptualize a gravitational or magnetic force that acts "
    output += (
        "on the grid tiles in each train input grid to pull them all to one side. "
    )
    output += (
        "Grid tiles can not pass through each other so they must stack against each "
    )
    output += "other in when they encounter another tile that is closer to the side. "
    output += "I've tested this candidate transformation on each of the train pairs, and it works for all of them. "
    output += "Therefore, I've applied the candidate transformation to the test input grid to get the following test output grid: "

    for i, (test_output_grid) in enumerate(test_output_grids):
        output += f"Test_Output_{i+1}=["
        for j in range(len(test_output_grid)):
            output += "["
            for k in range(len(test_output_grid[j])):
                output += str(test_output_grid[j][k])
                if k != len(test_output_grid[j]) - 1:
                    output += ","
            output += "]"
            if j != len(test_output_grid) - 1:
                output += ","
        output += "]"
        if i != len(test_output_grids) - 1:
            output += ", "
    output += "."

    # Tokenize the request text
    prompt = instruction + " " + output
    tokenized_request = tokenizer.tokenize(prompt)

    # Get the token length
    token_length = len(tokenized_request)

    return instruction, output, token_length


def create_isolate_obj_grids_prompt(
    min_grid_dim, max_grid_dim, num_train_tasks, num_test_tasks, tokenizer
):
    input_grids, output_grids = make_isolate_obj_grids(
        min_grid_dim, max_grid_dim, num_train_tasks, num_test_tasks
    )

    train_input_grids = input_grids[:num_train_tasks]
    train_output_grids = output_grids[:num_train_tasks]

    test_input_grids = input_grids[num_train_tasks:]
    test_output_grids = output_grids[num_train_tasks:]

    instruction = create_instruction(
        train_input_grids, train_output_grids, test_input_grids
    )

    output = "The candidate transformation is to isolate the sub-grid that is common "
    output += "to every train input grid and delete the other grid tiles (replace integer with symbolic integer 0). "
    output += "I've tested this candidate transformation on each of the train pairs, and it works for all of them. "
    output += "Therefore, I've applied the candidate transformation to the test input grid to get the following test output grid: "

    for i, (test_output_grid) in enumerate(test_output_grids):
        output += f"Test_Output_{i+1}=["
        for j in range(len(test_output_grid)):
            output += "["
            for k in range(len(test_output_grid[j])):
                output += str(test_output_grid[j][k])
                if k != len(test_output_grid[j]) - 1:
                    output += ","
            output += "]"
            if j != len(test_output_grid) - 1:
                output += ","
        output += "]"
        if i != len(test_output_grids) - 1:
            output += ", "
    output += "."

    # Tokenize the request text
    prompt = instruction + " " + output
    tokenized_request = tokenizer.tokenize(prompt)

    # Get the token length
    token_length = len(tokenized_request)

    return instruction, output, token_length


def create_largest_smallest_obj_grids_prompt(
    min_grid_dim, max_grid_dim, num_train_tasks, num_test_tasks, tokenizer
):
    keep_larger, input_grids, output_grids = make_largest_smallest_obj_grids(
        min_grid_dim, max_grid_dim, num_train_tasks, num_test_tasks
    )

    train_input_grids = input_grids[:num_train_tasks]
    train_output_grids = output_grids[:num_train_tasks]

    test_input_grids = input_grids[num_train_tasks:]
    test_output_grids = output_grids[num_train_tasks:]

    instruction = create_instruction(
        train_input_grids, train_output_grids, test_input_grids
    )

    output = "The candidate transformation to create an output grid that is "
    if keep_larger > 0:
        output += "the largest sub-grid in the input grid."
    else:
        output += "the smallest sub-grid in the input grid."
    output += "I've tested this candidate transformation on each of the train pairs, and it works for all of them. "
    output += "Therefore, I've applied the candidate transformation to the test input grid to get the following test output grid: "

    for i, (test_output_grid) in enumerate(test_output_grids):
        output += f"Test_Output_{i+1}=["
        for j in range(len(test_output_grid)):
            output += "["
            for k in range(len(test_output_grid[j])):
                output += str(test_output_grid[j][k])
                if k != len(test_output_grid[j]) - 1:
                    output += ","
            output += "]"
            if j != len(test_output_grid) - 1:
                output += ","
        output += "]"
        if i != len(test_output_grids) - 1:
            output += ", "
    output += "."

    # Tokenize the request text
    prompt = instruction + " " + output
    tokenized_request = tokenizer.tokenize(prompt)

    # Get the token length
    token_length = len(tokenized_request)

    return instruction, output, token_length


def create_mask_obj_grids_prompt(
    min_grid_dim, max_grid_dim, num_train_tasks, num_test_tasks, tokenizer
):
    mask_color, input_grids, output_grids = make_mask_obj_data_grids(
        min_grid_dim, max_grid_dim, num_train_tasks, num_test_tasks
    )

    train_input_grids = input_grids[:num_train_tasks]
    train_output_grids = output_grids[:num_train_tasks]

    test_input_grids = input_grids[num_train_tasks:]
    test_output_grids = output_grids[num_train_tasks:]

    instruction = create_instruction(
        train_input_grids, train_output_grids, test_input_grids
    )

    output = "The candidate transformation is to "
    if mask_color > 0:
        output += "replace the sub-grid that is common "
        output += (
            "to every train input grid with the grid tiles of the color (symbolic integer "
            + str(mask_color)
            + ") "
        )
        output += "used to replace the sub-grids in every train output grid. "
    else:
        output += (
            "delete (replace with symbolic integer 0) the sub-grid that is common "
        )
        output += "to every train input grid. "

    output += "I've tested this candidate transformation on each of the train pairs, and it works for all of them. "
    output += "Therefore, I've applied the candidate transformation to the test input grid to get the following test output grid: "

    for i, (test_output_grid) in enumerate(test_output_grids):
        output += f"Test_Output_{i+1}=["
        for j in range(len(test_output_grid)):
            output += "["
            for k in range(len(test_output_grid[j])):
                output += str(test_output_grid[j][k])
                if k != len(test_output_grid[j]) - 1:
                    output += ","
            output += "]"
            if j != len(test_output_grid) - 1:
                output += ","
        output += "]"
        if i != len(test_output_grids) - 1:
            output += ", "
    output += "."

    # Tokenize the request text
    prompt = instruction + " " + output
    tokenized_request = tokenizer.tokenize(prompt)

    # Get the token length
    token_length = len(tokenized_request)

    return instruction, output, token_length


def create_mirror_grids_prompt(
    min_grid_dim, max_grid_dim, num_train_tasks, num_test_tasks, tokenizer
):
    flip_type, input_grids, output_grids = make_mirror_grids(
        min_grid_dim, max_grid_dim, num_train_tasks, num_test_tasks
    )

    train_input_grids = input_grids[:num_train_tasks]
    train_output_grids = output_grids[:num_train_tasks]

    test_input_grids = input_grids[num_train_tasks:]
    test_output_grids = output_grids[num_train_tasks:]

    instruction = create_instruction(
        train_input_grids, train_output_grids, test_input_grids
    )

    output = "The candidate transformation is to flip the entire train input grid "
    if flip_type == 0:
        output += "vertically. "
    else:
        output += "horizontally. "

    output += "I've tested this candidate transformation on each of the train pairs, and it works for all of them. "
    output += "Therefore, I've applied the candidate transformation to the test input grid to get the following test output grid: "

    for i, (test_output_grid) in enumerate(test_output_grids):
        output += f"Test_Output_{i+1}=["
        for j in range(len(test_output_grid)):
            output += "["
            for k in range(len(test_output_grid[j])):
                output += str(test_output_grid[j][k])
                if k != len(test_output_grid[j]) - 1:
                    output += ","
            output += "]"
            if j != len(test_output_grid) - 1:
                output += ","
        output += "]"
        if i != len(test_output_grids) - 1:
            output += ", "
    output += "."

    # Tokenize the request text
    prompt = instruction + " " + output
    tokenized_request = tokenizer.tokenize(prompt)

    # Get the token length
    token_length = len(tokenized_request)

    return instruction, output, token_length


def create_most_freq_in_row_grids_prompt(
    min_grid_dim, max_grid_dim, num_train_tasks, num_test_tasks, tokenizer
):
    find_column, input_grids, output_grids = make_most_freq_in_row_grids(
        min_grid_dim, max_grid_dim, num_train_tasks, num_test_tasks
    )

    train_input_grids = input_grids[:num_train_tasks]
    train_output_grids = output_grids[:num_train_tasks]

    test_input_grids = input_grids[num_train_tasks:]
    test_output_grids = output_grids[num_train_tasks:]

    instruction = create_instruction(
        train_input_grids, train_output_grids, test_input_grids
    )

    output = "The candidate transformation to create a 1 "
    if find_column == 0:
        output += "column "
    else:
        output += "row "

    output += "output grid with the most common color in the input grid's "
    if find_column == 0:
        output += "row. "
    else:
        output += "column. "

    output += "I've tested this candidate transformation on each of the train pairs, and it works for all of them. "
    output += "Therefore, I've applied the candidate transformation to the test input grid to get the following test output grid: "

    for i, (test_output_grid) in enumerate(test_output_grids):
        output += f"Test_Output_{i+1}=["
        for j in range(len(test_output_grid)):
            output += "["
            for k in range(len(test_output_grid[j])):
                output += str(test_output_grid[j][k])
                if k != len(test_output_grid[j]) - 1:
                    output += ","
            output += "]"
            if j != len(test_output_grid) - 1:
                output += ","
        output += "]"
        if i != len(test_output_grids) - 1:
            output += ", "
    output += "."

    # Tokenize the request text
    prompt = instruction + " " + output
    tokenized_request = tokenizer.tokenize(prompt)

    # Get the token length
    token_length = len(tokenized_request)

    return instruction, output, token_length


def create_multiple_objs_grids_prompt(
    min_grid_dim, max_grid_dim, num_train_tasks, num_test_tasks, tokenizer
):
    input_grids, output_grids = make_multiple_objs_grids(
        min_grid_dim, max_grid_dim, num_train_tasks, num_test_tasks
    )

    train_input_grids = input_grids[:num_train_tasks]
    train_output_grids = output_grids[:num_train_tasks]

    test_input_grids = input_grids[num_train_tasks:]
    test_output_grids = output_grids[num_train_tasks:]

    instruction = create_instruction(
        train_input_grids, train_output_grids, test_input_grids
    )

    output = "The candidate transformation is to isolate the sub-grids that exist in multiples in each train input grid "
    output += "by deleting (replacing with symbolic integer 0) the other grid tiles. "
    output += "I've tested this candidate transformation on each of the train pairs, and it works for all of them. "
    output += "Therefore, I've applied the candidate transformation to the test input grid to get the following test output grid: "

    for i, (test_output_grid) in enumerate(test_output_grids):
        output += f"Test_Output_{i+1}=["
        for j in range(len(test_output_grid)):
            output += "["
            for k in range(len(test_output_grid[j])):
                output += str(test_output_grid[j][k])
                if k != len(test_output_grid[j]) - 1:
                    output += ","
            output += "]"
            if j != len(test_output_grid) - 1:
                output += ","
        output += "]"
        if i != len(test_output_grids) - 1:
            output += ", "
    output += "."

    # Tokenize the request text
    prompt = instruction + " " + output
    tokenized_request = tokenizer.tokenize(prompt)

    # Get the token length
    token_length = len(tokenized_request)

    return instruction, output, token_length


def create_rays_grids_prompt(
    min_grid_dim, max_grid_dim, num_train_tasks, num_test_tasks, tokenizer
):
    input_grids, output_grids = make_rays_grids(
        min_grid_dim, max_grid_dim, num_train_tasks, num_test_tasks
    )

    train_input_grids = input_grids[:num_train_tasks]
    train_output_grids = output_grids[:num_train_tasks]

    test_input_grids = input_grids[num_train_tasks:]
    test_output_grids = output_grids[num_train_tasks:]

    instruction = create_instruction(
        train_input_grids, train_output_grids, test_input_grids
    )

    output = "The candidate transformation is that a ray is extended from any grid tile on edge of the train input grid. "
    output += "The ray is a striaght line of grid tilesthe same color as the initial grid tile. "
    output += (
        "This ray extends until it hits another grid tile or the edge of the grid. "
    )
    output += "I've tested this candidate transformation on each of the train pairs, and it works for all of them. "
    output += "Therefore, I've applied the candidate transformation to the test input grid to get the following test output grid: "

    for i, (test_output_grid) in enumerate(test_output_grids):
        output += f"Test_Output_{i+1}=["
        for j in range(len(test_output_grid)):
            output += "["
            for k in range(len(test_output_grid[j])):
                output += str(test_output_grid[j][k])
                if k != len(test_output_grid[j]) - 1:
                    output += ","
            output += "]"
            if j != len(test_output_grid) - 1:
                output += ","
        output += "]"
        if i != len(test_output_grids) - 1:
            output += ", "
    output += "."

    # Tokenize the request text
    prompt = instruction + " " + output
    tokenized_request = tokenizer.tokenize(prompt)

    # Get the token length
    token_length = len(tokenized_request)

    return instruction, output, token_length


def create_rotate_grids_prompt(
    min_grid_dim, max_grid_dim, num_train_tasks, num_test_tasks, tokenizer
):
    rot_num, input_grids, output_grids = make_rotate_grids(
        min_grid_dim, max_grid_dim, num_train_tasks, num_test_tasks
    )

    train_input_grids = input_grids[:num_train_tasks]
    train_output_grids = output_grids[:num_train_tasks]

    test_input_grids = input_grids[num_train_tasks:]
    test_output_grids = output_grids[num_train_tasks:]

    instruction = create_instruction(
        train_input_grids, train_output_grids, test_input_grids
    )

    output = (
        "The candidate transformation is to rotate the entire train input grid "
        + str(rot_num * 90)
        + " degrees clockwise. "
    )

    output += "I've tested this candidate transformation on each of the train pairs, and it works for all of them. "
    output += "Therefore, I've applied the candidate transformation to the test input grid to get the following test output grid: "

    for i, (test_output_grid) in enumerate(test_output_grids):
        output += f"Test_Output_{i+1}=["
        for j in range(len(test_output_grid)):
            output += "["
            for k in range(len(test_output_grid[j])):
                output += str(test_output_grid[j][k])
                if k != len(test_output_grid[j]) - 1:
                    output += ","
            output += "]"
            if j != len(test_output_grid) - 1:
                output += ","
        output += "]"
        if i != len(test_output_grids) - 1:
            output += ", "
    output += "."

    # Tokenize the request text
    prompt = instruction + " " + output
    tokenized_request = tokenizer.tokenize(prompt)

    # Get the token length
    token_length = len(tokenized_request)

    return instruction, output, token_length


def create_same_color_objs_grids_prompt(
    min_grid_dim, max_grid_dim, num_train_tasks, num_test_tasks, tokenizer
):
    keep_same, input_grids, output_grids = make_same_color_objs_grids(
        min_grid_dim, max_grid_dim, num_train_tasks, num_test_tasks
    )

    train_input_grids = input_grids[:num_train_tasks]
    train_output_grids = output_grids[:num_train_tasks]

    test_input_grids = input_grids[num_train_tasks:]
    test_output_grids = output_grids[num_train_tasks:]

    instruction = create_instruction(
        train_input_grids, train_output_grids, test_input_grids
    )

    output = "The candidate transformation is that sub-grids in each train input grid that are "
    if keep_same > 0:
        output += "not the same colors are removed to get the corresponding train output grid. "
    else:
        output += (
            "the same colors are removed to get the corresponding train output grid. "
        )

    output += "I've tested this candidate transformation on each of the train pairs, and it works for all of them. "
    output += "Therefore, I've applied the candidate transformation to the test input grid to get the following test output grid: "

    for i, (test_output_grid) in enumerate(test_output_grids):
        output += f"Test_Output_{i+1}=["
        for j in range(len(test_output_grid)):
            output += "["
            for k in range(len(test_output_grid[j])):
                output += str(test_output_grid[j][k])
                if k != len(test_output_grid[j]) - 1:
                    output += ","
            output += "]"
            if j != len(test_output_grid) - 1:
                output += ","
        output += "]"
        if i != len(test_output_grids) - 1:
            output += ", "
    output += "."

    # Tokenize the request text
    prompt = instruction + " " + output
    tokenized_request = tokenizer.tokenize(prompt)

    # Get the token length
    token_length = len(tokenized_request)

    return instruction, output, token_length


###############################################################################
if __name__ == "__main__":
    min_grid_dim = 8
    max_grid_dim = 12
    num_train_tasks = randint(2, 3)
    num_test_tasks = 1
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")

    random_puzzle_type = randint(0, 18)
    # random_puzzle_type = 18
    if random_puzzle_type == 0:
        instruction, output, token_length = create_rotate_obj_puzzle_prompt(
            min_grid_dim, max_grid_dim, num_train_tasks, num_test_tasks, tokenizer
        )
    elif random_puzzle_type == 1:
        instruction, output, token_length = create_move_obj_puzzle_prompt(
            min_grid_dim, max_grid_dim, num_train_tasks, num_test_tasks, tokenizer
        )
    elif random_puzzle_type == 2:
        (
            instruction,
            output,
            token_length,
        ) = create_mirrored_obj_puzzle_prompt(
            min_grid_dim, max_grid_dim, num_train_tasks, num_test_tasks, tokenizer
        )
    elif random_puzzle_type == 3:
        instruction, output, token_length = create_scaled_obj_puzzle_prompt(
            min_grid_dim, max_grid_dim, num_train_tasks, num_test_tasks, tokenizer
        )
    elif random_puzzle_type == 4:
        (
            instruction,
            output,
            token_length,
        ) = create_swapped_color_grids_prompt(
            min_grid_dim, max_grid_dim, num_train_tasks, num_test_tasks, tokenizer
        )
    elif random_puzzle_type == 5:
        instruction, output, token_length = create_same_shape_grids_prompt(
            min_grid_dim, max_grid_dim, num_train_tasks, num_test_tasks, tokenizer
        )
    elif random_puzzle_type == 6:
        instruction, output, token_length = create_fill_pattern_holes_grids_prompt(
            min_grid_dim, max_grid_dim, num_train_tasks, num_test_tasks, tokenizer
        )
    elif random_puzzle_type == 7:
        (
            instruction,
            output,
            token_length,
        ) = create_fill_rotated_pattern_holes_grids_prompt(
            min_grid_dim, max_grid_dim, num_train_tasks, num_test_tasks, tokenizer
        )
    elif random_puzzle_type == 8:
        (
            instruction,
            output,
            token_length,
        ) = create_fill_surrounded_grids_prompt(
            min_grid_dim, max_grid_dim, num_train_tasks, num_test_tasks, tokenizer
        )
    elif random_puzzle_type == 9:
        (
            instruction,
            output,
            token_length,
        ) = create_gravity_grids_prompt(
            min_grid_dim, max_grid_dim, num_train_tasks, num_test_tasks, tokenizer
        )
    elif random_puzzle_type == 10:
        (
            instruction,
            output,
            token_length,
        ) = create_isolate_obj_grids_prompt(
            min_grid_dim, max_grid_dim, num_train_tasks, num_test_tasks, tokenizer
        )
    elif random_puzzle_type == 11:
        (
            instruction,
            output,
            token_length,
        ) = create_largest_smallest_obj_grids_prompt(
            min_grid_dim, max_grid_dim, num_train_tasks, num_test_tasks, tokenizer
        )
    elif random_puzzle_type == 12:
        (
            instruction,
            output,
            token_length,
        ) = create_mask_obj_grids_prompt(
            min_grid_dim, max_grid_dim, num_train_tasks, num_test_tasks, tokenizer
        )
    elif random_puzzle_type == 13:
        (
            instruction,
            output,
            token_length,
        ) = create_mirror_grids_prompt(
            min_grid_dim, max_grid_dim, num_train_tasks, num_test_tasks, tokenizer
        )
    elif random_puzzle_type == 14:
        (
            instruction,
            output,
            token_length,
        ) = create_most_freq_in_row_grids_prompt(
            min_grid_dim, max_grid_dim, num_train_tasks, num_test_tasks, tokenizer
        )
    elif random_puzzle_type == 15:
        (
            instruction,
            output,
            token_length,
        ) = create_multiple_objs_grids_prompt(
            min_grid_dim, max_grid_dim, num_train_tasks, num_test_tasks, tokenizer
        )
    elif random_puzzle_type == 16:
        (
            instruction,
            output,
            token_length,
        ) = create_rays_grids_prompt(
            min_grid_dim, max_grid_dim, num_train_tasks, num_test_tasks, tokenizer
        )
    elif random_puzzle_type == 17:
        (
            instruction,
            output,
            token_length,
        ) = create_rotate_grids_prompt(
            min_grid_dim, max_grid_dim, num_train_tasks, num_test_tasks, tokenizer
        )
    elif random_puzzle_type == 18:
        (
            instruction,
            output,
            token_length,
        ) = create_same_color_objs_grids_prompt(
            min_grid_dim, max_grid_dim, num_train_tasks, num_test_tasks, tokenizer
        )

    print("\ninstruction\n", instruction)
    print("\noutput\n", output)
    print("\ntoken_length", token_length)
