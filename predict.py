# Prediction interface for Cog ⚙️
# https://cog.run/python

from cog import BasePredictor, Input, Path
import torch
from PIL import Image
from typing import List
from transformers import AutoProcessor, AutoModelForVision2Seq

MODEL_CACHE = "checkpoints"

class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        # Initialize processor and model
        self.processor = AutoProcessor.from_pretrained(MODEL_CACHE)
        self.model = AutoModelForVision2Seq.from_pretrained(
            MODEL_CACHE,
            torch_dtype=torch.bfloat16,
            _attn_implementation="flash_attention_2",
        ).to("cuda")

    def predict(
        self,
        image: Path = Input(description="Input image to process"),
        prompt: str = Input(
            description="Text prompt to guide the model's response",
            default="Can you describe this image?"
        ),
        max_new_tokens: int = Input(
            description="Maximum number of tokens to generate",
            default=500,
            ge=1,
            le=2000,
        ),
    ) -> str:
        """Run a single prediction on the model"""
        # Load and process image
        pil_image = Image.open(image).convert('RGB')
        # Create input messages
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt}
                ]
            },
        ]
        # Prepare inputs
        model_prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(
            text=model_prompt,
            images=[pil_image],
            return_tensors="pt"
        )
        inputs = inputs.to("cuda")

        # Generate outputs
        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
        )
        generated_text = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
        )[0]
        
        # Extract only the Assistant's response
        if "Assistant:" in generated_text:
            response = generated_text.split("Assistant:")[-1].strip()
        else:
            response = generated_text.strip()
            
        return response
