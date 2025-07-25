import asyncio
import json
from typing import Dict, Any, Optional
from aiogram import types
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup

from sd_client import StableDiffusionClient

class AdvancedGenerationStates(StatesGroup):
    waiting_for_prompt = State()
    waiting_for_negative_prompt = State()
    waiting_for_steps = State()
    waiting_for_cfg_scale = State()
    waiting_for_size = State()

class AdvancedFeatures:
    def __init__(self, sd_client: StableDiffusionClient):
        self.sd_client = sd_client
    
    async def start_advanced_generation(self, message: types.Message, state: FSMContext):
        """Начать продвинутую генерацию с настройкой параметров"""
        await state.set_state(AdvancedGenerationStates.waiting_for_prompt)
        await message.answer(
            "🎨 Продвинутая генерация\n\n"
            "Отправьте основной промпт (описание изображения):"
        )
    
    async def handle_advanced_prompt(self, message: types.Message, state: FSMContext):
        """Обработка основного промпта"""
        prompt = message.text
        await state.update_data(prompt=prompt)
        await state.set_state(AdvancedGenerationStates.waiting_for_negative_prompt)
        
        await message.answer(
            "📝 Промпт сохранен!\n\n"
            "Теперь отправьте негативный промпт (что НЕ должно быть на изображении):\n"
            "Или отправьте 'skip' для пропуска"
        )
    
    async def handle_negative_prompt(self, message: types.Message, state: FSMContext):
        """Обработка негативного промпта"""
        negative_prompt = message.text if message.text.lower() != 'skip' else ""
        await state.update_data(negative_prompt=negative_prompt)
        await state.set_state(AdvancedGenerationStates.waiting_for_steps)
        
        await message.answer(
            "📝 Негативный промпт сохранен!\n\n"
            "Отправьте количество шагов (20-50, по умолчанию 20):\n"
            "Или отправьте 'default' для значения по умолчанию"
        )
    
    async def handle_steps(self, message: types.Message, state: FSMContext):
        """Обработка количества шагов"""
        try:
            if message.text.lower() == 'default':
                steps = 20
            else:
                steps = int(message.text)
                if steps < 1 or steps > 100:
                    await message.answer("❌ Количество шагов должно быть от 1 до 100. Попробуйте снова:")
                    return
        except ValueError:
            await message.answer("❌ Введите число. Попробуйте снова:")
            return
        
        await state.update_data(steps=steps)
        await state.set_state(AdvancedGenerationStates.waiting_for_cfg_scale)
        
        await message.answer(
            f"📝 Шаги: {steps}\n\n"
            "Отправьте CFG Scale (1-20, по умолчанию 7):\n"
            "Или отправьте 'default' для значения по умолчанию"
        )
    
    async def handle_cfg_scale(self, message: types.Message, state: FSMContext):
        """Обработка CFG Scale"""
        try:
            if message.text.lower() == 'default':
                cfg_scale = 7
            else:
                cfg_scale = float(message.text)
                if cfg_scale < 1 or cfg_scale > 20:
                    await message.answer("❌ CFG Scale должен быть от 1 до 20. Попробуйте снова:")
                    return
        except ValueError:
            await message.answer("❌ Введите число. Попробуйте снова:")
            return
        
        await state.update_data(cfg_scale=cfg_scale)
        await state.set_state(AdvancedGenerationStates.waiting_for_size)
        
        await message.answer(
            f"📝 CFG Scale: {cfg_scale}\n\n"
            "Отправьте размер изображения в формате 'ширинаxвысота':\n"
            "Например: 512x512, 768x512, 512x768\n"
            "Или отправьте 'default' для 512x512"
        )
    
    async def handle_size(self, message: types.Message, state: FSMContext):
        """Обработка размера изображения"""
        if message.text.lower() == 'default':
            width, height = 512, 512
        else:
            try:
                size_parts = message.text.lower().split('x')
                if len(size_parts) != 2:
                    raise ValueError
                width = int(size_parts[0])
                height = int(size_parts[1])
                
                # Проверяем, что размеры кратные 8 (требование SD)
                if width % 8 != 0 or height % 8 != 0:
                    await message.answer("❌ Размеры должны быть кратны 8. Попробуйте снова:")
                    return
                
                if width < 64 or height < 64 or width > 2048 or height > 2048:
                    await message.answer("❌ Размеры должны быть от 64 до 2048. Попробуйте снова:")
                    return
                    
            except (ValueError, IndexError):
                await message.answer("❌ Неверный формат. Используйте 'ширинаxвысота'. Попробуйте снова:")
                return
        
        await state.update_data(width=width, height=height)
        
        # Получаем все сохраненные данные
        data = await state.get_data()
        
        # Показываем итоговые параметры
        summary = f"""
🎨 Параметры генерации:

📝 Промпт: {data['prompt']}
🚫 Негативный промпт: {data.get('negative_prompt', 'Не указан')}
👣 Шаги: {data['steps']}
⚖️ CFG Scale: {data['cfg_scale']}
📏 Размер: {data['width']}x{data['height']}

Генерирую изображение...
        """
        
        status_msg = await message.answer(summary)
        
        try:
            # Генерируем изображение с кастомными параметрами
            result = self.sd_client.txt2img(
                prompt=data['prompt'],
                negative_prompt=data.get('negative_prompt', ''),
                steps=data['steps'],
                cfg_scale=data['cfg_scale'],
                width=data['width'],
                height=data['height']
            )
            
            if result and 'images' in result:
                import base64
                import io
                
                # Декодируем изображение
                image_data = base64.b64decode(result['images'][0])
                image_bytes = io.BytesIO(image_data)
                
                # Отправляем изображение
                await message.bot.send_photo(
                    chat_id=message.chat.id,
                    photo=types.BufferedInputFile(image_bytes.getvalue(), filename="advanced_generated.png"),
                    caption=f"🎨 Продвинутая генерация\n\n📝 Промпт: {data['prompt']}"
                )
                
                await status_msg.delete()
            else:
                await status_msg.edit_text("❌ Ошибка при генерации изображения")
                
        except Exception as e:
            await status_msg.edit_text(f"❌ Произошла ошибка: {str(e)}")
        
        finally:
            await state.clear()
    
    async def show_samplers(self, message: types.Message):
        """Показать доступные сэмплеры"""
        samplers = [
            "Euler", "Euler a", "LMS", "Heun", "DPM2", "DPM2 a", "DPM++ 2S a",
            "DPM++ 2M", "DPM++ SDE", "DPM fast", "DPM adaptive", "LMS Karras",
            "DPM2 Karras", "DPM2 a Karras", "DPM++ 2S a Karras", "DPM++ 2M Karras",
            "DPM++ SDE Karras", "DDIM", "PLMS"
        ]
        
        sampler_list = "\n".join([f"• {sampler}" for sampler in samplers])
        
        await message.answer(
            f"🎲 Доступные сэмплеры:\n\n{sampler_list}\n\n"
            "По умолчанию используется: DPM++ 2M Karras"
        )
    
    async def show_models(self, message: types.Message):
        """Показать доступные модели с подробной информацией"""
        models = self.sd_client.get_models()
        if models:
            model_info = []
            for i, model in enumerate(models[:5], 1):  # Показываем первые 5
                title = model.get('title', 'Неизвестно')
                model_info.append(f"{i}. {title}")
            
            if len(models) > 5:
                model_info.append(f"... и еще {len(models) - 5} моделей")
            
            model_list = "\n".join(model_info)
            
            await message.answer(
                f"🤖 Доступные модели:\n\n{model_list}\n\n"
                "Используйте команду /switch_model для смены модели"
            )
        else:
            await message.answer("❌ Не удалось получить список моделей")
    
    async def switch_model(self, message: types.Message):
        """Сменить модель"""
        # Извлекаем название модели из сообщения
        text = message.text
        if text.startswith('/switch_model'):
            model_name = text.replace('/switch_model', '').strip()
            
            if not model_name:
                await message.answer(
                    "❌ Укажите название модели!\n\n"
                    "Пример: /switch_model novaFurryXL_illustriousV9b.safetensors"
                )
                return
            
            status_msg = await message.answer(f"🔄 Переключаю модель на: {model_name}")
            
            if self.sd_client.switch_model(model_name):
                await status_msg.edit_text(f"✅ Модель успешно переключена на: {model_name}")
            else:
                await status_msg.edit_text(f"❌ Ошибка при переключении модели: {model_name}")
        else:
            await message.answer(
                "❌ Неверный формат команды!\n\n"
                "Пример: /switch_model novaFurryXL_illustriousV9b.safetensors"
            ) 