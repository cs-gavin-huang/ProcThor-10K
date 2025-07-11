import os
import requests
import base64
import ollama
from typing import Optional
# import sglang as sgl
from PIL import Image

class VLMClient:
    def chat_completion(self, prompt, image_path=None, **kwargs):
        raise NotImplementedError

class OpenAIClient(VLMClient):
    def __init__(self, api_base, model_id, api_key_env):
        self.api_base = api_base
        self.model_id = model_id
        self.api_key = os.environ[api_key_env]

    def chat_completion(self, prompt, image_path=None, temperature=0.7, max_tokens=100):
        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt}
            ]
        }]
        if image_path:
            with open(image_path, "rb") as f:
                img_b64 = base64.b64encode(f.read()).decode()
            messages[0]["content"].append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{img_b64}"}
            })
        payload = {
            "model": self.model_id,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        resp = requests.post(f"{self.api_base}/chat/completions", json=payload, headers=headers)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()

# class TGIClient(VLMClient):
#     def __init__(self, api_base, model_id):
#         self.api_base = api_base
#         self.model_id = model_id
# 
#     def chat_completion(self, prompt, image_path=None, temperature=0.7, max_tokens=100):
#         messages = [{
#             "role": "user",
#             "content": [
#                 {"type": "text", "text": prompt}
#             ]
#         }]
#         if image_path:
#             with open(image_path, "rb") as f:
#                 img_b64 = base64.b64encode(f.read()).decode()
#             messages[0]["content"].append({
#                 "type": "image_url",
#                 "image_url": {"url": f"data:image/png;base64,{img_b64}"}
#             })
#         payload = {
#             "model": self.model_id,
#             "messages": messages,
#             "temperature": temperature,
#             "max_tokens": max_tokens
#         }
#         resp = requests.post(f"{self.api_base}/chat/completions", json=payload)
#         resp.raise_for_status()
#         return resp.json()["choices"][0]["message"]["content"].strip() 
# 
# class VLLMClient(VLMClient):
#     def __init__(self, model_id, api_base="http://localhost:30000"):
#         self.model_id = model_id
#         self.client = sgl.OpenAI(api_base=api_base, api_key="EMPTY")
# 
#     def chat_completion(self, prompt: str, image_path: Optional[str] = None, **kwargs) -> str:
#         try:
#             messages = [{"role": "user", "content": []}]
#             
#             content = [{"type": "text", "text": prompt}]
# 
#             if image_path:
#                 if not os.path.exists(image_path):
#                     print(f"DEBUG: VLLMClient: Image path does not exist: {image_path}")
#                     return "Error: Image path does not exist for VLLM."
#                 
#                 # sglang can take image path directly
#                 content.append({"type": "image_url", "image_url": {"url": image_path}})
# 
#             messages[0]["content"] = content
# 
#             # Get temperature from kwargs or use a default
#             temperature = kwargs.get('temperature', 0.7)
#             # Get max_tokens from kwargs or use a default
#             max_tokens = kwargs.get('max_tokens', 100)
#             
#             print(f"DEBUG: VLLMClient: Sending request to model '{self.model_id}'. Temperature: {temperature}, Max Tokens: {max_tokens}. Prompt: '{prompt[:100]}...'")
# 
#             response = self.client.chat.completions.create(
#                 model=self.model_id,
#                 messages=messages,
#                 temperature=temperature,
#                 max_tokens=max_tokens,
#             )
# 
#             return response.choices[0].message.content.strip()
# 
#         except Exception as e:
#             error_message = f"Unexpected error in VLLMClient.chat_completion for model '{self.model_id}': {e}."
#             if image_path:
#                 error_message += f" Original image path provided: {image_path}"
#             print(f"DEBUG: {error_message}")
#             return error_message

class OllamaClient(VLMClient):
    def __init__(self, model_id: str, **kwargs):
        self.model_name = model_id
        self.client = ollama.Client(**kwargs)

    def chat_completion(self, prompt: str, image_path: Optional[str] = None, **kwargs) -> str:
        try:
            processed_prompt = prompt
            if image_path:
                if not os.path.exists(image_path):
                    print(f"DEBUG: OllamaClient: Image path does not exist: {image_path}")
                    return "Error: Image path does not exist for Ollama."

                abs_image_path = os.path.abspath(image_path)
                # Append the image path to the prompt, similar to CLI usage
                processed_prompt = f"{prompt} {abs_image_path}"
                print(f"DEBUG: OllamaClient: Appended image path to prompt. New prompt: '{processed_prompt[:100]}...'")
            # else: # No specific message needed if no image_path, handled by later debug print
                # print(f"DEBUG: OllamaClient: No image path provided.")

            options = {}
            if 'temperature' in kwargs:
                options['temperature'] = kwargs['temperature']
            if 'max_tokens' in kwargs:
                options['num_predict'] = kwargs['max_tokens']
            if 'keep_alive' in kwargs: # Added to handle keep_alive
                options['keep_alive'] = kwargs['keep_alive']

            image_info_for_debug = f"Image path '{image_path}' included in prompt." if image_path else "No image path provided."
            print(f"DEBUG: OllamaClient: Sending request to model '{self.model_name}' using ollama.generate. {image_info_for_debug} Options: {options}. Prompt: '{processed_prompt[:100]}...'")

            # Call ollama.generate without the explicit 'images' parameter.
            # The Ollama server is expected to parse the image path from the prompt string.
            response = self.client.generate(
                model=self.model_name,
                prompt=processed_prompt, # Prompt now contains the image path
                options=options if options else None,
                stream=False
            )
            
            if response and "response" in response:
                # If keep_alive was set to 0 for unload, the response content for "unload" prompt is not critical.
                if kwargs.get('keep_alive') == 0 and prompt == "unload":
                     print(f"DEBUG: OllamaClient: Model {self.model_name} received unload signal (keep_alive=0).")
                     return "Unload signal sent." # Or some other indicator
                return response["response"].strip()
            else:
                print(f"DEBUG: OllamaClient: Unexpected response structure from ollama.generate: {response}")
                return "Error: Unexpected response structure from Ollama generate."

        except ollama.ResponseError as e:
            error_message = f"Ollama API Error (generate) for model '{self.model_name}': {e.error} (status code: {e.status_code})."
            if image_path and not (kwargs.get('keep_alive') == 0 and prompt == "unload"): # Don't include image path details for unload prompt
                error_message += f" Original image path provided: {image_path}"
            print(f"DEBUG: {error_message}")
            return error_message
        except Exception as e:
            error_message = f"Unexpected error in OllamaClient.chat_completion (using generate with path in prompt) for model '{self.model_name}': {e}."
            if image_path and not (kwargs.get('keep_alive') == 0 and prompt == "unload"):
                error_message += f" Original image path provided: {image_path}"
            print(f"DEBUG: {error_message}")
            return error_message 