import os, shutil
import sys
sys.path.append("../OmniGen")
from OmniGen import OmniGenPipeline

def clear_all_in_folder():
    import os, shutil
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

def generate_multiple(model, checkpoint="0020000",it_number=3):
    clear_all_in_folder()

    #modify for your inference
    training_name = "snappr_finetunning_2"
    checkpoint = "0020000"

    image_path = "static"
    ref_image_name = "test_image.png"

    save_path =  image_path

    ### Lora configurations:
    if False:
        pipe = OmniGenPipeline.from_pretrained("Shitao/OmniGen-v1")  
        pipe.merge_lora(os.path.join("/home/ec2-user/snappr/snappr_analysis/results",training_name,"checkpoints",checkpoint))  

        for i in range(it_number):
            ## Multi-modal to Image
            images = pipe(
                prompt="<img><|image_1|></img> make this more appealing, refine the details to make it look more realistic and attractive for the view",
                input_images=[os.path.join(image_path,ref_image_name)],
                height=1024, 
                width=1024,
                guidance_scale=2.5, 
                img_guidance_scale=1.6,
                seed=i
            )
            images[0].save(os.path.join(save_path,f"inference-{i}"))  # save output PIL image
    else:
        print("fake saving pictures")
        return