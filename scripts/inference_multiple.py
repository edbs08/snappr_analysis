import os
import sys
sys.path.append("../OmniGen")
from OmniGen import OmniGenPipeline

#modify for your inference
training_name = "snappr_finetunning_2"
checkpoint = "0020000"

image_path = "/home/ec2-user/snappr/snappr_analysis/data/images/media/"
save_path =  os.path.join("/home/ec2-user/snappr/snappr_analysis/image_results",training_name+checkpoint+"multiple")
# Create the output folder if it doesn't exist
if not os.path.exists(save_path):
    os.makedirs(save_path)

img_test_list = [
    "54bf2448-ac14-44d2-8f35-0a485bf75cfe.png", #"output_image":"cc36f1da-b95d-42b1-822e-c6f08db08345.png"
    "1919a326-d2df-45bb-8ddd-5147997557b8.png", #"output_image":"666e2df7-8d9d-412b-9299-2561a8e3ab02.png"
    "f24fe8c5-a385-4651-8dd8-c74c8f6f64b7.png", #"output_image":"e8e1412e-70e8-45a0-a8d2-997fa21c38eb.png"
]
it_number = 3

pipe = OmniGenPipeline.from_pretrained("Shitao/OmniGen-v1")  
pipe.merge_lora(os.path.join("/home/ec2-user/snappr/snappr_analysis/results",training_name,"checkpoints",checkpoint))  

for i in range(it_number):
    for img in img_test_list:
        ## Multi-modal to Image
        images = pipe(
            prompt="<img><|image_1|></img> make this more appealing, refine the details to make it look more realistic and attractive for the view",
            input_images=[os.path.join(image_path,img)],
            height=1024, 
            width=1024,
            guidance_scale=2.5, 
            img_guidance_scale=1.6,
            seed=i
        )
        images[0].save(os.path.join(save_path,f"test-{i}-{img}"))  # save output PIL image
