import json
from PIL import Image, ImageTk
import tkinter as tk
import os

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

if __name__ == "__main__":
    #json file to use
    # jsonl_file = "C:\\Users\\edaeurb\\Documents\\snappr\\ai_utils\\data\\dataset_test.jsonl"  # Replace with your JSONL file path
    # jsonl_file = "C:\\Users\\edaeurb\\Documents\\snappr\\ai_utils\\data\\dataset_train.jsonl"  # Replace with your JSONL file path
    jsonl_file = "C:\\Users\\edaeurb\\Documents\\snappr\\ai_utils\\data\\dataset_validate.jsonl"  # Replace with your JSONL file path

    #image path 
    image_path = "C:\\Users\\edaeurb\\Documents\\snappr\\ai_utils\\data\\images\\media"
    viewer = ImageViewer(jsonl_file,image_path)
    viewer.run()