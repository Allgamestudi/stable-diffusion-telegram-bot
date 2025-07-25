"""
Модуль для автоматического улучшения промптов
"""

def enhance_prompt(prompt: str) -> str:
    """
    Улучшает промпт, добавляя качественные теги
    
    Args:
        prompt (str): Исходный промпт пользователя
        
    Returns:
        str: Улучшенный промпт с качественными тегами
    """
    quality_tags = "(masterpiece, best quality, 8k:1.3)"
    
    # Проверяем, есть ли уже качественные теги
    if any(tag in prompt.lower() for tag in ["masterpiece", "best quality", "8k"]):
        return prompt
    
    # Добавляем качественные теги в начало промпта
    return f"{quality_tags} {prompt}"

def get_default_negative_prompt() -> str:
    """
    Возвращает стандартный негативный промпт для улучшения качества
    
    Returns:
        str: Стандартный негативный промпт
    """
    return "(text:1.3), (deformed:1.3), (bad anatomy:1.4), (mutated paws:1.3), (lowres:1.2), (blurry:1.2), (censored:1.4)"

def get_enhanced_generation_params(prompt: str, custom_params: dict | None = None) -> dict:
    """
    Создает параметры для генерации с улучшенными промптами
    
    Args:
        prompt (str): Исходный промпт пользователя
        custom_params (dict | None): Дополнительные параметры
        
    Returns:
        dict: Параметры для генерации с улучшенными промптами
    """
    enhanced_prompt = enhance_prompt(prompt)
    negative_prompt = get_default_negative_prompt()
    
    # Базовые параметры
    params = {
        'prompt': enhanced_prompt,
        'negative_prompt': negative_prompt,
        'steps': 20,
        'sampler_name': 'DPM++ 2M Karras',
        'cfg_scale': 7,
        'width': 512,
        'height': 512,
        'batch_size': 1
    }
    
    # Добавляем пользовательские параметры
    if custom_params:
        params.update(custom_params)
    
    return params

def is_prompt_enhanced(prompt: str) -> bool:
    """
    Проверяет, содержит ли промпт качественные теги
    
    Args:
        prompt (str): Промпт для проверки
        
    Returns:
        bool: True если промпт уже содержит качественные теги
    """
    quality_tags = ["masterpiece", "best quality", "8k"]
    return any(tag in prompt.lower() for tag in quality_tags)

def get_prompt_info(prompt: str) -> dict:
    """
    Возвращает информацию о промпте
    
    Args:
        prompt (str): Промпт для анализа
        
    Returns:
        dict: Информация о промпте
    """
    enhanced = enhance_prompt(prompt)
    is_already_enhanced = is_prompt_enhanced(prompt)
    
    return {
        'original': prompt,
        'enhanced': enhanced,
        'is_already_enhanced': is_already_enhanced,
        'negative_prompt': get_default_negative_prompt(),
        'added_tags': '(masterpiece, best quality, 8k:1.3)' if not is_already_enhanced else 'Нет'
    } 