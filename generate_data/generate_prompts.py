import attr
import numpy as np
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

    rand_rot_or_flip = randint(0, 5)
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

            if rand_scale_or_shrink > 50:
                input_grids.append(input_grid)
                output_grids.append(output_grid)
            else:
                input_grids.append(output_grid)
                output_grids.append(input_grid)

            if rand_rot_or_flip > 0 and rand_rot_or_flip < 4:
                input_grid = np.rot90(input_grid, k=rand_rot_or_flip)
                output_grid = np.rot90(output_grid, k=rand_rot_or_flip)
            elif rand_rot_or_flip == 4:
                input_grid = np.fliplr(input_grid)
                output_grid = np.fliplr(output_grid)
            elif rand_rot_or_flip == 5:
                input_grid = np.flipud(input_grid)
                output_grid = np.flipud(output_grid)

            pair_count += 1

    if attempts == max_attempts:
        print("make_scaled_obj_grids: Failed to create grids after maximum attempts")
        return None, None, None

    return rand_scale_or_shrink, input_grids, output_grids


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
        while black_row_or_column or overlapping:
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
        obj_3_loop_count = 0
        while black_row_or_column or overlapping or same_as_sub_grid_1:
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


###################################################################################################################
def create_instruction(train_input_grids, train_output_grids, test_input_grids):
    instruction = "Given the following input/output train pairs of ARCSolver grids: "

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

    instruction += ". Find the transformation from each input grid to output grid that is common to all 3 train pairs."
    instruction += " Then apply this transformation to the following test input grid to get the test output grid: "

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

    train_input_grids = input_grids[:3]
    train_output_grids = output_grids[:3]

    test_input_grids = input_grids[3:]
    test_output_grids = output_grids[3:]

    instruction = create_instruction(
        train_input_grids, train_output_grids, test_input_grids
    )

    output = (
        "The common transformation is that the non-zero element sub-grid in each train input grid is rotated "
        + str(rot_num * 90)
    )

    if rand_rot_grid == 0 and rand_flip_grid == 0:
        # No grid transformation = rotate counter-clockwise around top-left corner
        output += " degrees counter-clockwise about its top left element to get the corresponding train output grid. Therefore, "
    elif rand_rot_grid == 0 and rand_flip_grid == 1:
        # Flip grid left-right, no grid rotation = rotate clockwise around top-right corner
        output += " degrees clockwise about its top right element to get the corresponding train output grid. Therefore, "
    elif rand_rot_grid == 1 and rand_flip_grid == 0:
        # Rotate grid 90 degrees counter-clockwise, no flip = rotate counter-clockwise around bottom-left corner
        output += " degrees counter-clockwise about its bottom left element to get the corresponding train output grid. Therefore, "
    elif rand_rot_grid == 1 and rand_flip_grid == 1:
        # Rotate grid 90 degrees counter-clockwise and flip left-right = rotate clockwise around bottom-right corner
        output += " degrees clockwise about its bottom right element to get the corresponding train output grid. Therefore, "
    elif rand_rot_grid == 2 and rand_flip_grid == 0:
        # Rotate grid 180 degrees counter-clockwise = rotate counter-clockwise around bottom-right corner
        output += " degrees counter-clockwise about its bottom right element to get the corresponding train output grid. Therefore, "
    elif rand_rot_grid == 2 and rand_flip_grid == 1:
        # Rotate grid 180 degrees counter-clockwise and flip left-right = rotate clockwise around bottom-left corner
        output += " degrees clockwise about its bottom left element to get the corresponding train output grid. Therefore, "
    elif rand_rot_grid == 3 and rand_flip_grid == 0:
        # Rotate grid 270 degrees counter-clockwise = rotate counter-clockwise around top-right corner
        output += " degrees counter-clockwise about its top right element to get the corresponding train output grid. Therefore, "
    else:
        # Rotate grid 270 degrees counter-clockwise and flip left-right = rotate clockwise around top-left corner
        output += " degrees clockwise about its top left element to get the corresponding train output grid. Therefore, "

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

    train_input_grids = input_grids[:3]
    train_output_grids = output_grids[:3]

    test_input_grids = input_grids[3:]
    test_output_grids = output_grids[3:]

    instruction = create_instruction(
        train_input_grids, train_output_grids, test_input_grids
    )

    output = (
        "The common transformation is that the non-zero element sub-grid in each train input grid is moved "
        + str(row_move)
        + " units vertically and "
        + str(column_move)
        + " units horizontally to get the corresponding train output grid. Therefore, "
    )
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

    train_input_grids = input_grids[:3]
    train_output_grids = output_grids[:3]

    test_input_grids = input_grids[3:]
    test_output_grids = output_grids[3:]

    instruction = create_instruction(
        train_input_grids, train_output_grids, test_input_grids
    )

    output = "The common transformation is that the non-zero element sub-grid in each train input grid is mirrored "
    if flip_type == 0:
        output += "vertically"
    else:
        output += "horizontally"
    output += (
        " about its center to get the corresponding train output grid. Therefore, "
    )
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
    rand_scale_or_shrink, input_grids, output_grids = make_scaled_obj_grids(
        min_grid_dim, max_grid_dim, num_train_tasks, num_test_tasks
    )

    train_input_grids = input_grids[:3]
    train_output_grids = output_grids[:3]

    test_input_grids = input_grids[3:]
    test_output_grids = output_grids[3:]

    instruction = create_instruction(
        train_input_grids, train_output_grids, test_input_grids
    )

    output = "The common transformation is that the non-zero element sub-grid in each train input grid is scaled "
    if rand_scale_or_shrink > 50:
        output += "up"
    else:
        output += "down"
    output += " by 2 about its top left element to get the corresponding train output grid. Therefore, "
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

    train_input_grids = input_grids[:3]
    train_output_grids = output_grids[:3]

    test_input_grids = input_grids[3:]
    test_output_grids = output_grids[3:]

    instruction = create_instruction(
        train_input_grids, train_output_grids, test_input_grids
    )

    output = "The common transformation is that certain integers in each train input grid are "
    output += "swapped with other integers to get the corresponding train output grid. "
    output += "These swapped integers are the same throughout all of the train sets. Therefore, "

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

    train_input_grids = input_grids[:3]
    train_output_grids = output_grids[:3]

    test_input_grids = input_grids[3:]
    test_output_grids = output_grids[3:]

    instruction = create_instruction(
        train_input_grids, train_output_grids, test_input_grids
    )

    output = (
        "The common transformation is that sub-grids in each train input grid that are "
    )
    if keep_same > 0:
        output += "not the same shape are removed to get the corresponding train output grid. Therefore, "
    else:
        output += "the same shape are removed to get the corresponding train output grid. Therefore, "

    # outputare the only sub-grids that remain in each train output grid. Therefore, "

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


if __name__ == "__main__":
    min_grid_dim = 6
    max_grid_dim = 30
    num_train_tasks = 3
    num_test_tasks = 1
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")

    random_puzzle_type = randint(0, 5)
    random_puzzle_type = 5
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
    else:
        instruction, output, token_length = create_same_shape_grids_prompt(
            min_grid_dim, max_grid_dim, num_train_tasks, num_test_tasks, tokenizer
        )

    print("\ninstruction\n", instruction)
    print("\noutput\n", output)
    print("\ntoken_length", token_length)
