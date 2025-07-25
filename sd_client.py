import requests
import base64
import json
from typing import Dict, Any, Optional
from config import SD_WEBUI_URL, DEFAULT_PARAMS

class StableDiffusionClient:
    def __init__(self, base_url: str = SD_WEBUI_URL):
        self.base_url = base_url.rstrip('/')
        
    def _make_request(self, endpoint: str, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Выполняет запрос к Stable Diffusion WebUI API"""
        try:
            url = f"{self.base_url}{endpoint}"
            response = requests.post(url, json=data, timeout=300)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Ошибка при запросе к SD WebUI: {e}")
            return None
    
    def txt2img(self, prompt: str, **kwargs) -> Optional[Dict[str, Any]]:
        """Генерирует изображение из текста"""
        # Объединяем параметры по умолчанию с переданными
        params = DEFAULT_PARAMS.copy()
        params.update(kwargs)
        params["prompt"] = prompt
        
        data = {
            "prompt": params["prompt"],
            "negative_prompt": params["negative_prompt"],
            "steps": params["steps"],
            "sampler_name": params["sampler_name"],
            "cfg_scale": params["cfg_scale"],
            "width": params["width"],
            "height": params["height"],
            "batch_size": params["batch_size"]
        }
        
        return self._make_request("/sdapi/v1/txt2img", data)
    
    def get_models(self) -> Optional[list]:
        """Получает список доступных моделей"""
        try:
            response = requests.get(f"{self.base_url}/sdapi/v1/sd-models", timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Ошибка при получении списка моделей: {e}")
            return None
    
    def switch_model(self, model_name: str) -> bool:
        """Переключает модель"""
        try:
            # Получаем список моделей для проверки
            models = self.get_models()
            if not models:
                print(f"Не удалось получить список моделей")
                return False
            
            # Ищем модель по имени
            model_found = False
            for model in models:
                if model.get('model_name') == model_name or model.get('title') == model_name:
                    model_found = True
                    break
            
            if not model_found:
                print(f"Модель '{model_name}' не найдена в списке доступных моделей")
                return False
            
            # Отправляем запрос на смену модели
            data = {"sd_model_checkpoint": model_name}
            result = self._make_request("/sdapi/v1/options", data)
            
            if result is not None:
                print(f"Модель успешно переключена на: {model_name}")
                return True
            else:
                print(f"Ошибка при переключении модели на: {model_name}")
                return False
                
        except Exception as e:
            print(f"Исключение при переключении модели: {e}")
            return False
    
    def is_available(self) -> bool:
        """Проверяет доступность SD WebUI"""
        try:
            response = requests.get(f"{self.base_url}/sdapi/v1/sd-models", timeout=5)
            return response.status_code == 200
        except:
            return False 