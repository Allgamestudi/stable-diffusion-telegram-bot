import os
from dotenv import load_dotenv

load_dotenv()

# Telegram Bot settings
BOT_TOKEN = os.getenv('BOT_TOKEN')

# Stable Diffusion settings
SD_WEBUI_URL = os.getenv('SD_WEBUI_URL', 'http://127.0.0.1:7860')
SD_MODEL_PATH = r"C:\Users\allga\stable-diffusion-webui\models\Stable-diffusion\novaFurryXL_illustriousV9b.safetensors"

# Default model settings
DEFAULT_MODEL = "novaFurryXL_illustriousV9b.safetensors"
DEFAULT_MODEL_TITLE = "novaFurryXL_illustriousV9b.safetensors"

# Default generation parameters
DEFAULT_PARAMS = {
    "prompt": "",
    "negative_prompt": "lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry",
    "steps": 20,
    "sampler_name": "DPM++ 2M Karras",
    "cfg_scale": 7,
    "width": 512,
    "height": 512,
    "batch_size": 1
} 