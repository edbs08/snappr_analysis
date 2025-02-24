import os
from PIL import Image

def resize_images(input_folder,output_folder,new_size):
    """
        function to resize png images in a folder
    """
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Loop through all files in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith('.png'):
            # Open the image
            img_path = os.path.join(input_folder, filename)
            img = Image.open(img_path)
            
            # Resize the image
            resized_img = img.resize(new_size, Image.Resampling.LANCZOS)
            
            # Save the resized image to the output folder
            output_path = os.path.join(output_folder, filename)
            resized_img.save(output_path)

            # print(f"Resized and saved: {filename}")

    print("All images have been resized and saved.")

if __name__ == "__main__":
    #### Resize images
    # Define the paths
    input_folder = 'C:\\Users\\edaeurb\\Documents\\snappr\\ai_utils\\data\\images\\media'
    output_folder = 'C:\\Users\\edaeurb\\Documents\\snappr\\ai_utils\\data\\images\\media2'
    # Define size
    new_size = (256,256)  # New resolution (width, height)
    # Call resize function
    resize_images(input_folder,output_folder,new_size)