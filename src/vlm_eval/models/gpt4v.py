from pathlib import Path
from .base import BaseVLM

class GPT4V(BaseVLM):
    def _load_model(self):
        from openai import OpenAI
        return OpenAI()
    
    def evaluate_image(self, image_path: Path, prompt: str) -> str:
        response = self.model.chat.completions.create(
            model="gpt-4-vision-preview",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{self._encode_image(image_path)}"}
                        }
                    ]
                }
            ],
            max_tokens=300
        )
        return response.choices[0].message.content.strip().lower() 