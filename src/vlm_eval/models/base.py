from abc import ABC, abstractmethod
from pathlib import Path

class BaseVLM(ABC):
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = self._load_model()
    
    @abstractmethod
    def _load_model(self):
        """加载模型"""
        pass
    
    @abstractmethod
    def evaluate_image(self, image_path: Path, prompt: str) -> str:
        """评估图片并返回结果"""
        pass
    
    def _encode_image(self, image_path: Path) -> str:
        """将图片编码为base64"""
        import base64
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8') 