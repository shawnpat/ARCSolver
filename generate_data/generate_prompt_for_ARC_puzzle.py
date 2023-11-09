import json
import os


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


def create_instruction(train_input_grids, train_output_grids, test_input_grids):
    instruction = "This is an abstract reasoning puzzle: "
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
    instruction += " Use your knowledge of core principals in physics, mathematics, and chain of thought reasoning to solve this puzzle. "
    instruction += "Step 1: Devise a candidate transformation and describe it in words. No coding or machine learning. Stop after you "
    instruction += "complete Step 1. "

    return instruction


###############################################################################
if __name__ == "__main__":
    file_path = "data/ARC/evaluation/0a2355a6.json"

    with open(file_path, "r") as f:
        data = json.load(f)
    train_tasks = data["train"]
    test_tasks = data["test"]

    train_input_grids = []
    train_output_grids = []
    test_input_grids = []
    test_output_grids = []

    for train_task in train_tasks:
        ti = train_task["input"]
        train_input_grids.append(ti)

        to = train_task["output"]
        train_output_grids.append(to)

    for test_task in test_tasks:
        ti = test_task["input"]
        test_input_grids.append(ti)

        to = test_task["output"]
        test_output_grids.append(to)

    instruction = create_instruction(
        train_input_grids, train_output_grids, test_input_grids
    )
    print("\ninstruction\n", instruction)
    print("\n")
    print("Apply the tranformation you've developed to the following test input grid: ")
    print(test_input_grids[0])
    print("Please predcit the test output grid:")
    print("\n")
    output = test_output_grids[0]
    print("\noutput\n", output)
