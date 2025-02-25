import os
from transformers import (
    CLIPTokenizer,
    CLIPTextModelWithProjection,
    CLIPVisionModelWithProjection,
    CLIPImageProcessor,
)
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
import numpy as np 

clip_id = "openai/clip-vit-large-patch14"
tokenizer = CLIPTokenizer.from_pretrained(clip_id)
text_encoder = CLIPTextModelWithProjection.from_pretrained(clip_id).to("cuda")
image_processor = CLIPImageProcessor.from_pretrained(clip_id)
image_encoder = CLIPVisionModelWithProjection.from_pretrained(clip_id).to("cuda")


class DirectionalSimilarity(nn.Module):
    def __init__(self, tokenizer, text_encoder, image_processor, image_encoder):
        super().__init__()
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder
        self.image_processor = image_processor
        self.image_encoder = image_encoder

    def preprocess_image(self, image):
        image = self.image_processor(image, return_tensors="pt")["pixel_values"]
        return {"pixel_values": image.to("cuda")}

    def tokenize_text(self, text):
        inputs = self.tokenizer(
            text,
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {"input_ids": inputs.input_ids.to("cuda")}

    def encode_image(self, image):
        preprocessed_image = self.preprocess_image(image)
        image_features = self.image_encoder(**preprocessed_image).image_embeds
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        return image_features

    def encode_text(self, text):
        tokenized_text = self.tokenize_text(text)
        text_features = self.text_encoder(**tokenized_text).text_embeds
        text_features = text_features / text_features.norm(dim=1, keepdim=True)
        return text_features

    def compute_directional_similarity(self, img_feat_one, img_feat_two, text_feat_one, text_feat_two):
        sim_direction = F.cosine_similarity(img_feat_two - img_feat_one, text_feat_two - text_feat_one)
        return sim_direction

    def forward(self, image_one, image_two, caption_one, caption_two):
        img_feat_one = self.encode_image(image_one)
        img_feat_two = self.encode_image(image_two)
        text_feat_one = self.encode_text(caption_one)
        text_feat_two = self.encode_text(caption_two)
        directional_similarity = self.compute_directional_similarity(
            img_feat_one, img_feat_two, text_feat_one, text_feat_two
        )
        return directional_similarity

def get_dir_similarity():
    dir_similarity = DirectionalSimilarity(tokenizer, text_encoder, image_processor, image_encoder)
    scores = []
    img_folder= "static"
    edited_images = ["inference-0.png",  "inference-1.png",  "inference-1.png"]
    original_image = "test_image.png"
    for edited_image in edited_images:

        original_caption = "a picture of a dish for a restaurant menu"
        modified_caption = "a picture of a dish for a restaurant menu with refined details and lightning"

        similarity_score = dir_similarity(
            Image.open(os.path.join(img_folder,original_image)), 
            Image.open(os.path.join(img_folder,edited_image)), 
            original_caption, 
            modified_caption
        )
        score = float(similarity_score.detach().cpu())

        print(f"""
            ***** CLI score original vs {edited_image} = {score}
              """)
        scores.append(score)

    print(f" \nAverage CLIP directional similarity: {np.mean(scores)}")
    print("The higher the CLIP directional similarity, the better it is.")