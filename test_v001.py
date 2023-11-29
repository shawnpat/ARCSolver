import sys
import json
import os
import re
import ast
import gc
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, PeftModel, PeftConfig
import time


def create_instruction(train_input_grids, train_output_grids, test_input_grids):
    instruction = ""
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

    test_task_count = 0
    for i, (test_input_grid) in enumerate(test_input_grids):
        test_task_count += 1
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

    for i in range(test_task_count):
        instruction += f" Test_Output_{i+1}=?"
        if i < test_task_count - 1:
            instruction += ", "

    return instruction


def load_and_print_file(file_path, gdrive_path):
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "caching_allocator"
    device_map = {"": 0}
    loop_start_time = time.time()

    # """
    # Load the specified file and print its contents.
    # """
    # try:
    #     with open(file_path, "r") as file:
    #         content = file.read()
    #         print(f"Contents of {file_path}:\n{content}")
    # except FileNotFoundError:
    #     print(f"File {file_path} not found.")
    # except Exception as e:
    #     print(f"An error occurred: {e}")

    gc.collect()
    torch.cuda.empty_cache()
    gc.collect()

    # if ".json" not in name:
    #   continue

    print(file_path)
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

    peft_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.1,
        r=64,
        bias="none",
        task_type="CAUSAL_LM",
    )

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=False,
    )

    peft_model_id = (
        gdrive_path + "adapters/Mistral-7B-finetuned_on_20000_minimal_puzzles"
    )

    config = PeftConfig.from_pretrained(peft_model_id)
    model = AutoModelForCausalLM.from_pretrained(
        config.base_model_name_or_path,
        quantization_config=bnb_config,
        return_dict=True,
        load_in_4bit=True,
        device_map={"": 0},
    )

    tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token

    # Load the Lora model
    model = PeftModel.from_pretrained(model, peft_model_id)

    query = "<s>[INST] " + instruction + " [/INST] "
    encodeds = tokenizer(query, return_tensors="pt", add_special_tokens=False)
    model_inputs = encodeds.to("cuda")

    model.eval()
    with torch.no_grad():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=1024,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    decoded = tokenizer.batch_decode(generated_ids)
    result = decoded[0]
    result = result[len(query) :]

    match = re.search(r"Test_Output_1=\[\[.*?\]\]", result)
    if match:
        array_string = match.group(0).replace("Test_Output_1=", "")
        array = ast.literal_eval(array_string)
        print("Result")
        print(result)
        print("")
    else:
        print("Result is nonsense")
        print(result)
        print("")
        array = []

    ground_truth = test_output_grids[0]

    print("Ground Truth")
    for row in ground_truth:
        print(row)
    print("Prediction")
    for row in array:
        print(row)
    if len(array) < 1:
        print("Prediction is nonsense")

    if array == ground_truth:
        print("...................................Correct!!!!!!!!!!!!!\n")
        num_correct += 1
    else:
        print("...................................Incorrect.\n")

    del peft_config
    del bnb_config
    del peft_model_id
    del config
    del model
    del tokenizer
    del query
    del encodeds
    del model_inputs
    del generated_ids
    del decoded
    del result
    del array
    del ground_truth
    del train_input_grids
    del train_output_grids
    del test_input_grids
    del test_output_grids
    del train_tasks
    del test_tasks
    del instruction
    del file_path
    del data
    gc.collect()
    torch.cuda.empty_cache()
    gc.collect()
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = ""

    loop_end_time = time.time()
    loop_execution_time = loop_end_time - loop_start_time
    print(f"This inference took {loop_execution_time} seconds to complete.")


def main():
    gdrive_path = "/workspace/"

    if len(sys.argv) != 2:
        print("Usage: python your_script.py <file_path>")
        sys.exit(1)

    file_path = sys.argv[1]
    load_and_print_file(file_path, gdrive_path)


if __name__ == "__main__":
    main()
