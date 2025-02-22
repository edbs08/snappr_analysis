import requests
import base64
from openai import OpenAI

class LLMConnection:
    def __init__(self, openai=True,model=None):
        API_KEY ="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"        
        self.openai_key = API_KEY
    
    def encode_image(self,image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
    
    def create_instruction_per_image(self,image):
        client = OpenAI(api_key=self.openai_key)
        # Getting the Base64 string
        base64_image_1 = self.encode_image(image)

        # Two step call to llm 
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": """this picture is a convination of a photography with a AI processing. 
                            tell me in one sentence what is missing to be more professional and appealing for a restaurant menu. maybe details on the igredients, color saturation, presence of more elements.
                            be very direct, no introductions, only the instruction. 
                            """,
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{base64_image_1}"},
                        },
                    ],
                }
            ],
        )
        return response.choices[0].message.content

if __name__=="__main__":
    llm = LLMConnection()
    llm.create_instruction_per_image("reference_img.png")


