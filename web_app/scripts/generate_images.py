import os, shutil
import sys
from PIL import Image

sys.path.append("../OmniGen")
from OmniGen import OmniGenPipeline

def clear_all_in_folder():
    folder = 'static'
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

def resize_and_save_image(image_path, resolution=(1024, 1024)):
    """
    Reads an image, resizes it to a specified symmetric resolution, and saves it as a PNG with '_mod' appended to the original name.

    :param image_path: Path to the input image.
    :param resolution: Tuple (width, height) representing the desired resolution. Default is (1024, 1024).
    """
    try:
        # Open the image
        with Image.open(image_path) as img:
            # Resize the image
            resized_img = img.resize(resolution, Image.Resampling.LANCZOS)
            
            # # Create the new file name
            # base_name, ext = os.path.splitext(image_path)
            # new_image_path = f"{base_name}_mod.png"
            
            # Save the resized image as PNG
            resized_img.save(image_path, "PNG")
            print(f"Image saved as {image_path}")
    
    except Exception as e:
        print(f"An error occurred: {e}")

def generate_multiple(model, checkpoint,instruction,resolution,it_number=3):

    image_path = "static"

    if instruction == "default":
        instruction = "make this more appealing, refine the details to make it look more realistic and attractive for the view"

    ref_image_name = "test_image.png"

    save_path =  image_path

    pipe = OmniGenPipeline.from_pretrained("Shitao/OmniGen-v1")  
    if model == "default":
        print("using original OmniGen-v1 model")
    else:
        print(f"Adding model {model} to pipeline")
        pipe.merge_lora(os.path.join("/home/ec2-user/snappr/snappr_analysis/results",model,"checkpoints",checkpoint))  

    for i in range(it_number):
        ## Multi-modal to Image
        images = pipe(
            prompt=f"<img><|image_1|></img> {instruction}",
            input_images=[os.path.join(image_path,ref_image_name)],
            height=resolution, 
            width=resolution,
            guidance_scale=2.5, 
            img_guidance_scale=3,
            seed=i
        )
        images[0].save(os.path.join(save_path,f"inference-{i}.png"))  # save output PIL image
