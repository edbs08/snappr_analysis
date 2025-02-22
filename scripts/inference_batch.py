import os
from OmniGen import OmniGenPipeline

#modify for your inference
training_name = "snappr_finetunning_2"
checkpoint = "0020000"

image_path = "/home/ec2-user/snappr/data/images/media/"
save_path =  os.path.join("/home/ec2-user/snappr/image_results",training_name+checkpoint)
# Create the output folder if it doesn't exist
if not os.path.exists(save_path):
    os.makedirs(save_path)

img_test_list = [
    "86b98016-9f5a-45a2-97ec-d13ea3e31db9.png", #"output_image":"566e916a-aa27-4a44-bc8d-b1b31c9c55f6.png"
    "54bf2448-ac14-44d2-8f35-0a485bf75cfe.png", #"output_image":"cc36f1da-b95d-42b1-822e-c6f08db08345.png"
    "1919a326-d2df-45bb-8ddd-5147997557b8.png", #"output_image":"666e2df7-8d9d-412b-9299-2561a8e3ab02.png"
    "f24fe8c5-a385-4651-8dd8-c74c8f6f64b7.png", #"output_image":"e8e1412e-70e8-45a0-a8d2-997fa21c38eb.png"
    "ab05b485-04e9-4522-8a33-8cbb6781fe38.png", #"output_image":"270e5b0a-79d3-47c6-b642-8a572495ec44.png"
]

pipe = OmniGenPipeline.from_pretrained("Shitao/OmniGen-v1")  
pipe.merge_lora(os.path.join("/home/ec2-user/snappr/results",training_name,"checkpoints",checkpoint))  

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
    images[0].save(os.path.join(save_path,f"test-{img}"))  # save output PIL image
