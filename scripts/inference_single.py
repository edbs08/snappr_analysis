import os
import sys
sys.path.append("../OmniGen")
from OmniGen import OmniGenPipeline

#modify for your inference
training_name = "snappr_finetunning_2"
checkpoint = "0020000"

image_path = "/home/ec2-user/snappr/snappr_analysis/data/images/media/"
save_path =  os.path.join("/home/ec2-user/snappr/snappr_analysis/image_results",training_name+checkpoint)
# Create the output folder if it doesn't exist
if not os.path.exists(save_path):
    os.makedirs(save_path)

img_test_list = [
    "86b98016-9f5a-45a2-97ec-d13ea3e31db9.png", #"output_image":"566e916a-aa27-4a44-bc8d-b1b31c9c55f6.png"
]

pipe = OmniGenPipeline.from_pretrained("Shitao/OmniGen-v1")  
pipe.merge_lora(os.path.join("/home/ec2-user/snappr/snappr_analysis/results",training_name,"checkpoints",checkpoint))  

for img in img_test_list:
    ## Multi-modal to Image
    images = pipe(
        prompt="<img><|image_1|></img> make this more appealing, refine the details to make it look more realistic and attractive for the view",
        input_images=[os.path.join(image_path,img)],
        height=1024, 
        width=1024,
        guidance_scale=2.5, 
        img_guidance_scale=1.6,
        seed=0
    )
    images[0].save(os.path.join(save_path,f"single1-{img}"))  # save output PIL image

