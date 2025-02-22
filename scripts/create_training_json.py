import json
import os
from coding_project.call_llm import LLMConnection

def get_image_pairs(jsonl_file):
    image_pairs = []
    with open(jsonl_file, "r") as file:
        for line in file:
            data = json.loads(line)
            image_pairs.append((data["input_image"], data["output_image"]))
    return image_pairs


def add_edit_instruction(image_pairs, image_path,llm_generation=False):
    instruction = "make this more appealing, refine the details to make it look more realistic and attractive for the view"
    data = []

    for input_image, output_image in image_pairs:
        if llm_generation:
            print("not implemented")
        data.append({
           "task_type":"image_edit","instruction":f"<img><|image_1|></img> {instruction}","input_images":[os.path.join(image_path,input_image)],"output_image":os.path.join(output_image)
        })
    # Open a file in write mode
    with open('output.jsonl', 'w') as f:
        # Write each dictionary as a JSON string on a new line
        for item in data:
            f.write(json.dumps(item))

def create_training_specs(jsonl_file,image_path):
    image_pairs = get_image_pairs(jsonl_file)
    add_edit_instruction(image_pairs,image_path)

def create_llm_instruction(image_path,json_file):
    data = []
    image_pairs = get_image_pairs(json_file)
    i = 0
    for input_image, _ in image_pairs:
        llm = LLMConnection()
        response = llm.create_instruction_per_image(os.path.join(image_path,input_image))
        print(i,response)
        data.append({
           "image":input_image,"instruction":response
        })
        i+=1
        if i >5:
            break


    with open('json_instruction.jsonl', 'w') as f:
        # Write each dictionary as a JSON string on a new line
        for item in data:
            f.write(json.dumps(item))


if __name__ == "__main__":
    json_file = "C:\\Users\\edaeurb\\Documents\\snappr\\ai_utils\\data\\dataset_train.jsonl"
    image_path = "C:\\Users\\edaeurb\\Documents\\snappr\\ai_utils\\data\\images\\media"
    #In case llm should be used for the instructions 
    create_llm_instruction(image_path,json_file)

    # create_training_specs(json_file,image_path)
