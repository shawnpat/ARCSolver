import json

# Load the JSON file
file_path = "data/ARC/evaluation/0a1d4ef5.json"

# Open the file and read its contents
with open(file_path, "r") as file:
    data = json.load(file)


# Iterate over each top-level key and print the corresponding 'input' and 'output' 2D arrays
blah = ""
for key in data:
    for i, item in enumerate(data[key]):
        blah += key + "_input_" + str(i + 1) + "="
        for j, row in enumerate(item["input"]):
            if j == 0:
                blah += "["
            blah += str(row)
            if j == len(item["input"]) - 1:
                blah += "]"
            else:
                blah += ","

        blah += key + "_output_" + str(i + 1) + "="
        for j, row in enumerate(item["output"]):
            if j == 0:
                blah += "["
            blah += str(row)
            if j == len(item["input"]) - 1:
                blah += "]"
            else:
                blah += ","
print(blah)
