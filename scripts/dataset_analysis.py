import json
from PIL import Image, ImageTk
import tkinter as tk
import os
from call_llm import LLMConnection

class ImageViewer:
    def __init__(self, jsonl_file,image_path):
        self.jsonl_file = jsonl_file
        self.image_pairs = self.load_image_pairs()
        self.current_index = 0
        self.root = tk.Tk()
        self.root.title("Image Pair Viewer")

        # Create frames for each image and its label
        self.frame1 = tk.Frame(self.root)
        self.frame1.pack(side=tk.LEFT, padx=10, pady=10)
        self.frame2 = tk.Frame(self.root)
        self.frame2.pack(side=tk.RIGHT, padx=10, pady=10)
        
        # Labels for image names
        self.label_name1 = tk.Label(self.frame1, text="", font=("Arial", 12))
        self.label_name1.pack()
        self.label_name2 = tk.Label(self.frame2, text="", font=("Arial", 12))
        self.label_name2.pack()

        # Labels for images
        self.label_image1 = tk.Label(self.frame1)
        self.label_image1.pack()
        self.label_image2 = tk.Label(self.frame2)
        self.label_image2.pack()

        # Bind arrow keys for navigation
        self.root.bind("<Left>", self.previous_image_pair)
        self.root.bind("<Right>", self.next_image_pair)

        #Define image path
        self.image_path = image_path
        self.show_image_pair()

    def load_image_pairs(self):
        image_pairs = []
        with open(self.jsonl_file, 'r') as file:
            for line in file:
                data = json.loads(line)
                image_pairs.append((data['input_image'], data['output_image']))
        return image_pairs

    def show_image_pair(self):
        input_image_path, output_image_path = self.image_pairs[self.current_index]

        # Display image names
        self.label_name1.config(text=input_image_path)
        self.label_name2.config(text=output_image_path)
        # Print names in console for reference
        print(f"input image: {input_image_path}")
        print(f"Output image: {output_image_path}")
        print("***")


        input_image = Image.open(os.path.join(self.image_path,input_image_path))
        output_image = Image.open(os.path.join(self.image_path,output_image_path))
        input_image.thumbnail((400, 400))
        output_image.thumbnail((400, 400))
        input_photo = ImageTk.PhotoImage(input_image)
        output_photo = ImageTk.PhotoImage(output_image)
        self.label_image1.config(image=input_photo)
        self.label_image1.image = input_photo
        self.label_image2.config(image=output_photo)
        self.label_image2.image = output_photo

    def next_image_pair(self, event=None):
        self.current_index = (self.current_index + 1) % len(self.image_pairs)
        self.show_image_pair()

    def previous_image_pair(self, event=None):
        self.current_index = (self.current_index - 1) % len(self.image_pairs)
        self.show_image_pair()

    def run(self):
        self.root.mainloop()

def check_classes_in_dataset(json_file,image_path):
    prompt = """
    Help me clasify this image withing the following classes. 
        pizza_like - shape of a pizza, bread based, can contain toppings. e.g. Pizza, Focaccia, Naan bread 
        dessert_like - sweet pastry can contain glazed e.g. muffin, brownie, cake, pancake
        sandwich_like - ingredients between one or two pieces of bread. e.g. sandwich, burger, tacos
        beverage - liquid in a container for drinking e.g. soda, coffee, juice
        soup_like - liquid, in a bowl. e.g. bean soup, tomato soup
        pasta - pasta based dish. e. g. spaghetti, mac and cheese, noodle 
        stuffed_dough - dough with some filling in the inside. e. g. empanada, dumplings, pasty, spring_rolls
        main_course - a dish with different elements. e. g. stake with fries, chicken with salad and side of bread
        salad - vegetables or fruit based dish. e. g. fruit salad, lettuce with veggies 
        side_dish - small portion, single element, not many elements. e.g. fries, rice, 
        condiment - portion of a single condiment. e. g. dressing, ketchup, mustard 
        other_xxxx - write it followed by what you think is the correct class. e.g. other_sushi, other_fried_chicken 

        if one element belongs to more than one class. pick the one more representative
        reply only with the word. do not add any other information or description, reply only the word of the class you chose for the image.
    """
    image_list = []
    with open(json_file, "r") as file:
        for line in file:
            data = json.loads(line)
            image_list.append(data["input_image"])
    classes_dict = {}

    for img in image_list:
        llm = LLMConnection()
        res = llm.general_vlm_call(prompt,os.path.join(image_path,img))
        #check for short responses
        if len(res)>25:
            break
        if res in classes_dict:
            classes_dict[res]+=1
        else:
            classes_dict[res]=1
        print(classes_dict)

if __name__ == "__main__":
    #image path 
    image_path = "C:\\Users\\edaeurb\\Documents\\snappr\\snappr_analysis\\data\\images\\media"
    
    #json file to use
    jsonl_file = "C:\\Users\\edaeurb\\Documents\\snappr\\snappr_analysis\\data\\dataset_test.jsonl"  # Replace with your JSONL file path
    # jsonl_file = "C:\\Users\\edaeurb\\Documents\\snappr\\snappr_analysis\\data\\dataset_train.jsonl"  # Replace with your JSONL file path
    # jsonl_file = "C:\\Users\\edaeurb\\Documents\\snappr\\snappr_analysis\\data\\dataset_validate.jsonl"  # Replace with your JSONL file path
    # check_classes_in_dataset(jsonl_file,image_path)

    ##### Dataset viewer
    viewer = ImageViewer(jsonl_file,image_path)
    viewer.run()