import asyncio
import base64
import io
import logging
import time
import threading
from concurrent.futures import ThreadPoolExecutor
from aiogram import Bot, Dispatcher, types, F
from aiogram.filters import Command
from aiogram.types import FSInputFile, InlineKeyboardMarkup, InlineKeyboardButton, ReplyKeyboardMarkup, KeyboardButton
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.fsm.storage.memory import MemoryStorage
from datetime import datetime

from config import BOT_TOKEN, SD_MODEL_PATH
from sd_client import StableDiffusionClient
from advanced_features import AdvancedFeatures, AdvancedGenerationStates
from queue_manager import queue_manager, GenerationStatus, GenerationStage
from prompt_enhancer import enhance_prompt, get_default_negative_prompt
import config

# Настройка логирования
logging.basicConfig(level=logging.INFO)

# Функции логирования
def log_user_message(message: types.Message):
    """Логирование сообщений пользователей"""
    user = message.from_user
    if user:
        log_msg = (
            f"[USER MSG] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | "
            f"user_id={user.id} | username={user.username or 'None'} | first_name={user.first_name or 'None'} | "
            f"text={repr(message.text)}"
        )
        logging.info(log_msg)
    else:
        logging.warning(f"[USER MSG] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Unknown user | text={repr(message.text)}")

def log_user_callback(callback: types.CallbackQuery):
    """Логирование callback запросов пользователей"""
    user = callback.from_user
    if user:
        log_msg = (
            f"[USER CALLBACK] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | "
            f"user_id={user.id} | username={user.username or 'None'} | first_name={user.first_name or 'None'} | "
            f"data={repr(callback.data)}"
        )
        logging.info(log_msg)
    else:
        logging.warning(f"[USER CALLBACK] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Unknown user | data={repr(callback.data)}")

# Инициализация бота и диспетчера
if not BOT_TOKEN:
    raise ValueError("BOT_TOKEN не установлен в config.py")
bot = Bot(token=BOT_TOKEN)
storage = MemoryStorage()
dp = Dispatcher(storage=storage)

# Инициализация клиентов
sd_client = StableDiffusionClient()
advanced_features = AdvancedFeatures(sd_client)

# Создаем пул потоков для обработки генерации
generation_executor = ThreadPoolExecutor(max_workers=3, thread_name_prefix="SD_Generator")

# Словарь для отслеживания активных задач
active_tasks = {}

# Состояния FSM
class GenerationStates(StatesGroup):
    waiting_for_prompt = State()
    waiting_for_params = State()
    waiting_for_animal_type = State()
    waiting_for_gender = State()
    waiting_for_fur_color = State()
    waiting_for_clothing = State()
    waiting_for_pose = State()
    waiting_for_expression = State()
    waiting_for_location = State()
    waiting_for_activity = State()
    waiting_for_priority = State()

# Создание клавиатур
def get_main_keyboard():
    """Создает основную клавиатуру"""
    keyboard = ReplyKeyboardMarkup(
        keyboard=[
            [
                KeyboardButton(text="🎨 Создать изображение"),
                KeyboardButton(text="🐾 Простая генерация")
            ],
            [
                KeyboardButton(text="🔄 Продвинутая генерация"),
                KeyboardButton(text="📊 Статус SD")
            ],
            [
                KeyboardButton(text="📋 Доступные модели"),
                KeyboardButton(text="🔄 Сменить модель")
            ],
            [
                KeyboardButton(text="🎲 Сэмплеры"),
                KeyboardButton(text="📋 Мои задачи")
            ],
            [
                KeyboardButton(text="📊 Очередь"),
                KeyboardButton(text="❓ Помощь")
            ],
            [
                KeyboardButton(text="⚙️ Настройки")
            ]
        ],
        resize_keyboard=True,
        input_field_placeholder="Или просто напишите описание изображения..."
    )
    return keyboard

def get_generation_keyboard(task_id: str):
    """Создает клавиатуру во время генерации"""
    keyboard = InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(text="⏹️ Отменить", callback_data=f"cancel_{task_id}")
            ]
        ]
    )
    return keyboard

def get_simple_generation_keyboard():
    """Создает клавиатуру для простой генерации"""
    keyboard = InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(text="🔙 Назад", callback_data="back_to_main"),
                InlineKeyboardButton(text="❌ Отменить", callback_data="cancel_simple")
            ]
        ]
    )
    return keyboard

def get_advanced_keyboard():
    """Создает клавиатуру для продвинутых функций"""
    keyboard = InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(text="👣 Настроить шаги", callback_data="settings_steps"),
                InlineKeyboardButton(text="⚖️ Настроить CFG", callback_data="settings_cfg")
            ],
            [
                InlineKeyboardButton(text="📏 Настроить размер", callback_data="settings_size"),
                InlineKeyboardButton(text="🎲 Выбрать сэмплер", callback_data="settings_sampler")
            ],
            [
                InlineKeyboardButton(text="🚫 Негативный промпт", callback_data="settings_negative"),
                InlineKeyboardButton(text="💾 Сохранить настройки", callback_data="settings_save")
            ],
            [
                InlineKeyboardButton(text="🔙 Назад", callback_data="back_to_main")
            ]
        ]
    )
    return keyboard

def get_models_keyboard():
    """Создает клавиатуру для выбора моделей"""
    try:
        models = sd_client.get_models()
        if not models:
            # Если не удалось получить модели, показываем стандартную
            keyboard = InlineKeyboardMarkup(
                inline_keyboard=[
                    [
                        InlineKeyboardButton(
                            text=f"⭐ {config.DEFAULT_MODEL_TITLE} (стандартная)",
                            callback_data=f"switch_model_{config.DEFAULT_MODEL}"
                        )
                    ],
                    [
                        InlineKeyboardButton(text="🔙 Назад", callback_data="back_to_main")
                    ]
                ]
            )
            return keyboard
        
        # Создаем кнопки для каждой модели
        buttons = []
        for model in models:
            model_title = model.get('title', model.get('model_name', 'Неизвестная модель'))
            model_name = model.get('model_name', model_title)
            
            # Отмечаем стандартную модель звездочкой
            if config.DEFAULT_MODEL in model_name or config.DEFAULT_MODEL in model_title:
                button_text = f"⭐ {model_title} (стандартная)"
            else:
                button_text = model_title
            
            # Используем model_name для callback_data, так как это точное имя файла
            buttons.append([
                InlineKeyboardButton(
                    text=button_text,
                    callback_data=f"switch_model_{model_name}"
                )
            ])
        
        # Добавляем кнопку "Назад"
        buttons.append([
            InlineKeyboardButton(text="🔙 Назад", callback_data="back_to_main")
        ])
        
        keyboard = InlineKeyboardMarkup(inline_keyboard=buttons)
        return keyboard
        
    except Exception as e:
        logging.error(f"Ошибка при создании клавиатуры моделей: {e}")
        # Возвращаем простую клавиатуру с ошибкой
        keyboard = InlineKeyboardMarkup(
            inline_keyboard=[
                [
                    InlineKeyboardButton(text="❌ Ошибка загрузки моделей", callback_data="back_to_main")
                ],
                [
                    InlineKeyboardButton(text="🔙 Назад", callback_data="back_to_main")
                ]
            ]
        )
        return keyboard


# Функции для работы с очередью
def get_stage_description(stage: GenerationStage) -> str:
    """Получает описание этапа генерации"""
    descriptions = {
        GenerationStage.INITIALIZING: "🔧 Инициализация...",
        GenerationStage.LOADING_MODEL: "🤖 Загрузка модели...",
        GenerationStage.PROCESSING_PROMPT: "📝 Обработка промпта...",
        GenerationStage.GENERATING_IMAGE: "🎨 Генерация изображения...",
        GenerationStage.ENCODING_RESULT: "💾 Кодирование результата...",
        GenerationStage.FINALIZING: "✨ Финальная обработка..."
    }
    return descriptions.get(stage, "⏳ Обработка...")

def get_stage_progress_range(stage: GenerationStage) -> tuple:
    """Получает диапазон прогресса для этапа"""
    ranges = {
        GenerationStage.INITIALIZING: (0, 10),
        GenerationStage.LOADING_MODEL: (10, 25),
        GenerationStage.PROCESSING_PROMPT: (25, 35),
        GenerationStage.GENERATING_IMAGE: (35, 85),
        GenerationStage.ENCODING_RESULT: (85, 95),
        GenerationStage.FINALIZING: (95, 100)
    }
    return ranges.get(stage, (0, 100))

async def process_generation_queue():
    """Обработчик очереди генерации с многопоточностью"""
    while True:
        try:
            # Начинаем обработку следующей задачи
            task = queue_manager.start_processing()
            if task:
                # Запускаем обработку задачи в отдельном потоке
                if task.id not in active_tasks:
                    active_tasks[task.id] = True
                    asyncio.create_task(process_task_async(task))
            else:
                # Нет задач в очереди, ждем
                await asyncio.sleep(1)
        except Exception as e:
            logging.error(f"Ошибка в обработке очереди: {e}")
            await asyncio.sleep(5)

async def process_task_async(task):
    """Асинхронная обработка задачи с многопоточностью"""
    try:
        # Запускаем генерацию в отдельном потоке
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(generation_executor, process_task_sync, task)
        
        if result:
            # Отправляем результат пользователю
            await send_generation_result(task, result)
        else:
            await send_generation_error(task, "Ошибка при генерации изображения")
            
    except Exception as e:
        logging.error(f"Ошибка при обработке задачи {task.id}: {e}")
        queue_manager.fail_task(task.id, str(e))
        await send_generation_error(task, str(e))
    finally:
        # Удаляем задачу из активных
        active_tasks.pop(task.id, None)

def process_task_sync(task):
    """Синхронная обработка задачи в отдельном потоке"""
    try:
        # Этап 1: Инициализация
        queue_manager.update_task_progress(task.id, GenerationStage.INITIALIZING, 5)
        time.sleep(0.5)
        
        # Этап 2: Загрузка модели
        queue_manager.update_task_progress(task.id, GenerationStage.LOADING_MODEL, 15)
        time.sleep(1)
        
        # Этап 3: Обработка промпта
        queue_manager.update_task_progress(task.id, GenerationStage.PROCESSING_PROMPT, 30)
        time.sleep(0.5)
        
        # Улучшаем промпт автоматически
        enhanced_prompt = enhance_prompt(task.prompt)
        enhanced_negative = get_default_negative_prompt()
        
        # Этап 4: Генерация изображения
        queue_manager.update_task_progress(task.id, GenerationStage.GENERATING_IMAGE, 40)
        
        # Симуляция прогресса генерации
        for progress in range(40, 85, 5):
            queue_manager.update_task_progress(task.id, GenerationStage.GENERATING_IMAGE, progress)
            time.sleep(0.3)
        
        # Подготавливаем параметры с улучшенными промптами
        generation_params = task.parameters or {}
        generation_params['prompt'] = enhanced_prompt
        generation_params['negative_prompt'] = enhanced_negative
        
        # Генерируем изображение
        result = sd_client.txt2img(enhanced_prompt, negative_prompt=enhanced_negative, **(task.parameters or {}))
        
        if result and 'images' in result:
            # Этап 5: Кодирование результата
            queue_manager.update_task_progress(task.id, GenerationStage.ENCODING_RESULT, 90)
            time.sleep(0.5)
            
            # Этап 6: Финальная обработка
            queue_manager.update_task_progress(task.id, GenerationStage.FINALIZING, 100)
            time.sleep(0.3)
            
            # Завершаем задачу
            queue_manager.complete_task(task.id, result)
            return result
        else:
            queue_manager.fail_task(task.id, "Ошибка при генерации изображения")
            return None
            
    except Exception as e:
        logging.error(f"Ошибка при обработке задачи {task.id}: {e}")
        queue_manager.fail_task(task.id, str(e))
        return None



async def send_generation_result(task, result):
    """Отправляет результат генерации пользователю"""
    try:
        # Декодируем изображение
        image_data = base64.b64decode(result['images'][0])
        image_bytes = io.BytesIO(image_data)
        
        # Получаем улучшенный промпт для отображения
        enhanced_prompt = enhance_prompt(task.prompt)
        negative_prompt = get_default_negative_prompt()
        
        # Отправляем изображение
        await bot.send_photo(
            chat_id=task.user_id,
            photo=types.BufferedInputFile(image_bytes.getvalue(), filename="generated.png"),
            caption=f"🎨 <b>Сгенерированное изображение</b>\n\n📝 Промпт: <code>{enhanced_prompt}</code>\n\n🚫 Негативный: <code>{negative_prompt}</code>\n\n✅ Задача завершена успешно!",
            reply_markup=get_main_keyboard(),
            parse_mode="HTML"
        )
    except Exception as e:
        logging.error(f"Ошибка при отправке результата: {e}")

async def send_generation_error(task, error):
    """Отправляет сообщение об ошибке пользователю"""
    try:
        # Получаем улучшенный промпт для отображения
        enhanced_prompt = enhance_prompt(task.prompt)
        negative_prompt = get_default_negative_prompt()
        
        await bot.send_message(
            chat_id=task.user_id,
            text=f"❌ <b>Ошибка генерации</b>\n\n📝 Промпт: <code>{enhanced_prompt}</code>\n\n🚫 Негативный: <code>{negative_prompt}</code>\n\n🚫 Ошибка: {error}",
            reply_markup=get_main_keyboard(),
            parse_mode="HTML"
        )
    except Exception as e:
        logging.error(f"Ошибка при отправке ошибки: {e}")

async def monitor_task_progress(task, status_msg):
    """Мониторинг прогресса задачи в фоне"""
    try:
        # Обновляем сообщение каждые 2 секунды во время ожидания в очереди
        while task.status == GenerationStatus.QUEUED:
            await asyncio.sleep(2)
            queue_position = queue_manager.get_queue_position(task.id)
            if queue_position > 0:
                await status_msg.edit_text(
                    f"📋 <b>Ожидание в очереди</b>\n\n"
                    f"📝 Промпт: <code>{task.prompt}</code>\n\n"
                    f"📊 Позиция в очереди: {queue_position}\n"
                    f"⏳ Ожидание обработки...",
                    reply_markup=get_generation_keyboard(task.id),
                    parse_mode="HTML"
                )
            else:
                break
        
        # Обновляем прогресс во время обработки
        while task.status == GenerationStatus.PROCESSING:
            await update_progress_message(task, status_msg)
            await asyncio.sleep(1)
            
        # Удаляем сообщение о прогрессе
        await status_msg.delete()
        
    except Exception as e:
        logging.error(f"Ошибка при мониторинге прогресса: {e}")

async def update_progress_message(task, message):
    """Обновляет сообщение с прогрессом"""
    try:
        stage_desc = get_stage_description(task.stage)
        progress_percent = int(task.progress)
        
        # Получаем позицию в очереди (если задача в очереди)
        queue_position = queue_manager.get_queue_position(task.id)
        queue_info = ""
        if queue_position > 0:
            queue_info = f"\n📋 Позиция в очереди: {queue_position}"
        
        status_text = f"""
🎨 <b>Генерация изображения...</b>

📝 Промпт: <code>{task.prompt}</code>
{stage_desc}
⏳ Прогресс: {progress_percent}%
{queue_info}
        """
        
        await message.edit_text(
            status_text,
            reply_markup=get_generation_keyboard(task.id),
            parse_mode="HTML"
        )
    except Exception as e:
        logging.error(f"Ошибка при обновлении прогресса: {e}")

# Команды
@dp.message(Command("start"))
async def cmd_start(message: types.Message):
    """Обработчик команды /start"""
    log_user_message(message)
    welcome_text = """
🎨 <b>Добро пожаловать в Advanced Stable Diffusion Bot!</b>

Используйте кнопки ниже или просто отправьте описание изображения!

💡 <b>Быстрый старт:</b>
• 🐾 <b>Простая генерация</b> - пошаговое создание персонажа
• 🎨 <b>Создать изображение</b> - прямая генерация по описанию
• Просто напишите что хотите создать

🔧 <b>Продвинутые возможности:</b>
• 🔄 Продвинутая генерация - полный контроль параметров
• 📊 Статус SD - проверить состояние WebUI
• 📋 Модели - список доступных моделей
• 🎲 Сэмплеры - выбор алгоритма генерации
• 🔄 Сменить модель - переключение между моделями
• 📋 Мои задачи - просмотр ваших задач
• 📊 Очередь - информация о очереди
• ⚙️ Настройки - настройка параметров по умолчанию
    """
    await message.answer(welcome_text, reply_markup=get_main_keyboard(), parse_mode="HTML")

@dp.message(Command("help"))
async def cmd_help(message: types.Message):
    """Обработчик команды /help"""
    log_user_message(message)
    help_text = """
📖 <b>Справка по использованию бота:</b>

🎨 <b>Создание изображений:</b>
• Нажмите "🎨 Создать изображение" или
• Просто отправьте текстовое описание

🔄 <b>Продвинутая генерация:</b>
• Полный контроль над всеми параметрами
• Настройка негативного промпта
• Выбор размера изображения
• Настройка количества шагов и CFG Scale

⚙️ <b>Параметры генерации:</b>
• 👣 Шаги: количество итераций (20 по умолчанию)
• ⚖️ CFG Scale: сила следования промпту (7 по умолчанию)
• 📏 Размер: размер изображения (512x512 по умолчанию)
• 🎲 Сэмплер: алгоритм генерации

📋 <b>Система очереди:</b>
• Задачи добавляются в очередь
• Отображается позиция в очереди
• Детальный прогресс по этапам
• Возможность отмены задач

📝 <b>Примеры промптов:</b>
• "красивая девушка в лесу, цифровое искусство"
• "космический корабль в стиле киберпанк"
• "милый котенок, акварель"
• "портрет рыцаря в доспехах, эпическое освещение"

✨ <b>Автоматическое улучшение:</b>
• К простым промптам автоматически добавляется "(masterpiece, best quality, 8k:1.3)"
• Автоматически добавляется негативный промпт для лучшего качества

🔄 <b>Управление моделями:</b>
• 📋 Модели - список доступных моделей
• 🔄 Сменить модель - переключение между моделями
• 🎲 Сэмплеры - список доступных алгоритмов
    """
    await message.answer(help_text, reply_markup=get_main_keyboard(), parse_mode="HTML")

@dp.message(Command("status"))
async def cmd_status(message: types.Message):
    """Проверка статуса SD WebUI"""
    log_user_message(message)
    status_msg = await message.answer("🔍 Проверяю статус Stable Diffusion WebUI...")
    
    if sd_client.is_available():
        models = sd_client.get_models()
        if models:
            current_model = next((m for m in models if m.get('title')), None)
            model_name = current_model.get('title', 'Неизвестно') if current_model else 'Неизвестно'
            
            # Получаем информацию о очереди
            queue_info = queue_manager.get_queue_info()
            
            status_text = f"""
✅ <b>Stable Diffusion WebUI доступен!</b>

🌐 URL: <code>{sd_client.base_url}</code>
🤖 Текущая модель: <code>{model_name}</code>
📊 Всего моделей: <code>{len(models)}</code>

📋 <b>Статистика очереди:</b>
• Задач в очереди: <code>{queue_info['queue_length']}</code>
• Обрабатывается: <code>{'Да' if queue_info['processing'] else 'Нет'}</code>
• Всего задач: <code>{queue_info['total_tasks']}</code>
• Завершено: <code>{queue_info['completed_tasks']}</code>
            """
        else:
            status_text = "✅ Stable Diffusion WebUI доступен, но не удалось получить список моделей"
    else:
        status_text = """
❌ <b>Stable Diffusion WebUI недоступен!</b>

Убедитесь, что:
1. SD WebUI запущен
2. API включен (--api флаг)
3. URL в конфигурации правильный
        """
    
    await status_msg.edit_text(status_text, parse_mode="HTML")

@dp.message(Command("models"))
async def cmd_models(message: types.Message):
    """Показать доступные модели"""
    log_user_message(message)
    await advanced_features.show_models(message)

@dp.message(Command("samplers"))
async def cmd_samplers(message: types.Message):
    """Показать доступные сэмплеры"""
    log_user_message(message)
    await advanced_features.show_samplers(message)

@dp.message(Command("switch_model"))
async def cmd_switch_model(message: types.Message):
    """Сменить модель"""
    log_user_message(message)
    await advanced_features.switch_model(message)

@dp.message(Command("generate"))
async def cmd_generate(message: types.Message, state: FSMContext):
    """Начать процесс генерации"""
    log_user_message(message)
    await state.set_state(GenerationStates.waiting_for_prompt)
    await message.answer("🎨 Отправьте описание изображения, которое хотите создать:")

@dp.message(Command("advanced"))
async def cmd_advanced(message: types.Message, state: FSMContext):
    """Начать продвинутую генерацию"""
    log_user_message(message)
    await advanced_features.start_advanced_generation(message, state)

# Обработка кнопок клавиатуры
@dp.message(F.text == "🎨 Создать изображение")
async def handle_create_image(message: types.Message, state: FSMContext):
    """Обработка кнопки создания изображения"""
    log_user_message(message)
    await state.set_state(GenerationStates.waiting_for_prompt)
    await message.answer(
        "🎨 <b>Создание изображения</b>\n\n"
        "Отправьте описание изображения, которое хотите создать.\n\n"
        "💡 <b>Примеры:</b>\n"
        "• beautiful girl in forest, digital art\n"
        "• space ship in cyberpunk style\n"
        "• cute kitten, watercolor\n\n"
        "⚠️ <b>Важно:</b> Пишите на английском языке!",
        reply_markup=get_simple_generation_keyboard(),
        parse_mode="HTML"
    )

@dp.message(F.text == "🐾 Простая генерация")
async def handle_simple_generation(message: types.Message, state: FSMContext):
    """Обработка кнопки простой генерации"""
    log_user_message(message)
    await state.set_state(GenerationStates.waiting_for_animal_type)
    await message.answer(
        "🐾 <b>Простая генерация персонажа</b>\n\n"
        "Давайте создадим персонажа пошагово!\n\n"
        "<b>Шаг 1/10:</b> Какой вид животного?\n\n"
        "💡 <b>Примеры:</b> fox, wolf, cat, dog, rabbit, bear, otter, deer, tiger, lion\n\n"
        "⚠️ <b>Важно:</b> Пишите на английском языке!",
        reply_markup=get_simple_generation_keyboard(),
        parse_mode="HTML"
    )

@dp.message(F.text == "🔄 Продвинутая генерация")
async def handle_advanced(message: types.Message, state: FSMContext):
    """Обработка кнопки продвинутой генерации"""
    log_user_message(message)
    await advanced_features.start_advanced_generation(message, state)

@dp.message(F.text == "📊 Статус SD")
async def handle_status(message: types.Message):
    """Обработка кнопки статуса"""
    log_user_message(message)
    await cmd_status(message)

@dp.message(F.text == "📋 Доступные модели")
async def handle_models(message: types.Message):
    """Обработка кнопки моделей"""
    log_user_message(message)
    await advanced_features.show_models(message)

@dp.message(F.text == "🎲 Сэмплеры")
async def handle_samplers(message: types.Message):
    """Обработка кнопки сэмплеров"""
    log_user_message(message)
    await advanced_features.show_samplers(message)

@dp.message(F.text == "🔄 Сменить модель")
async def handle_switch_model(message: types.Message):
    """Обработка кнопки смены модели"""
    log_user_message(message)
    # Получаем текущую модель
    current_model = "Неизвестно"
    try:
        models = sd_client.get_models()
        if models:
            current_model_obj = next((m for m in models if m.get('title')), None)
            if current_model_obj:
                current_model = current_model_obj.get('title', 'Неизвестно')
    except:
        pass
    
    await message.answer(
        f"🔄 <b>Выбор модели</b>\n\n"
        f"Выберите модель для переключения:\n\n"
        f"💡 <b>Советы:</b>\n"
        f"• ⭐ отмечена стандартная модель\n"
        f"• Переключение может занять несколько секунд\n"
        f"• Текущая модель: <code>{current_model}</code>",
        reply_markup=get_models_keyboard(),
        parse_mode="HTML"
    )

@dp.message(F.text == "📋 Мои задачи")
async def handle_my_tasks(message: types.Message):
    """Обработка кнопки моих задач"""
    log_user_message(message)
    user_tasks = queue_manager.get_user_tasks(message.from_user.id)
    
    if not user_tasks:
        await message.answer("📋 У вас пока нет задач.", reply_markup=get_main_keyboard())
        return
    
    tasks_text = "📋 <b>Ваши задачи:</b>\n\n"
    
    for i, task in enumerate(user_tasks[-5:], 1):  # Показываем последние 5 задач
        status_emoji = {
            GenerationStatus.QUEUED: "⏳",
            GenerationStatus.PROCESSING: "🔄",
            GenerationStatus.COMPLETED: "✅",
            GenerationStatus.FAILED: "❌",
            GenerationStatus.CANCELLED: "🚫"
        }.get(task.status, "❓")
        
        stage_desc = get_stage_description(task.stage) if task.status == GenerationStatus.PROCESSING else ""
        
        tasks_text += f"{i}. {status_emoji} <b>{task.prompt[:50]}...</b>\n"
        tasks_text += f"   Статус: {task.status.value}\n"
        if stage_desc:
            tasks_text += f"   {stage_desc}\n"
        tasks_text += "\n"
    
    await message.answer(tasks_text, reply_markup=get_main_keyboard(), parse_mode="HTML")

@dp.message(F.text == "📊 Очередь")
async def handle_queue_info(message: types.Message):
    """Обработка кнопки информации о очереди"""
    log_user_message(message)
    queue_info = queue_manager.get_queue_info()
    
    queue_text = f"""
📊 <b>Информация о очереди:</b>

📋 Задач в очереди: <code>{queue_info['queue_length']}</code>
🔄 Обрабатывается: <code>{'Да' if queue_info['processing'] else 'Нет'}</code>
📈 Всего задач: <code>{queue_info['total_tasks']}</code>
✅ Завершено: <code>{queue_info['completed_tasks']}</code>

💡 <b>Советы:</b>
• Задачи обрабатываются по очереди
• Можно отменить задачу во время генерации
• Используйте "Мои задачи" для просмотра статуса
    """
    
    await message.answer(queue_text, reply_markup=get_main_keyboard(), parse_mode="HTML")

@dp.message(F.text == "❓ Помощь")
async def handle_help(message: types.Message):
    """Обработка кнопки помощи"""
    log_user_message(message)
    await cmd_help(message)

@dp.message(F.text == "⚙️ Настройки")
async def handle_settings(message: types.Message):
    """Обработка кнопки настроек"""
    log_user_message(message)
    await message.answer("⚙️ <b>Настройки генерации:</b>\n\nВ разработке...", reply_markup=get_main_keyboard(), parse_mode="HTML")

# Обработка callback кнопок
@dp.callback_query(F.data == "back_to_main")
async def back_to_main(callback: types.CallbackQuery):
    """Возврат к главному меню"""
    log_user_callback(callback)
    if callback.message:
        await callback.message.edit_text("Главное меню")
        await callback.message.answer("Выберите действие:", reply_markup=get_main_keyboard())
    await callback.answer()

@dp.callback_query(F.data.startswith("switch_model_"))
async def handle_model_switch(callback: types.CallbackQuery):
    """Обработка смены модели"""
    log_user_callback(callback)
    if callback.data and callback.message:
        model_name = callback.data.replace("switch_model_", "")
        
        try:
            # Показываем сообщение о переключении
            await callback.message.edit_text(
                f"🔄 <b>Переключение модели...</b>\n\n"
                f"Модель: <code>{model_name}</code>\n"
                f"⏳ Пожалуйста, подождите...",
                parse_mode="HTML"
            )
            
            # Логируем попытку смены модели
            logging.info(f"Попытка смены модели на: {model_name}")
            
            # Переключаем модель
            success = sd_client.switch_model(model_name)
            
            if success:
                logging.info(f"Модель успешно переключена на: {model_name}")
                await callback.message.edit_text(
                    f"✅ <b>Модель успешно переключена!</b>\n\n"
                    f"Новая модель: <code>{model_name}</code>\n\n"
                    f"Теперь можно создавать изображения с новой моделью.",
                    parse_mode="HTML"
                )
                # Отправляем новое сообщение с главной клавиатурой
                await callback.message.answer("Выберите действие:", reply_markup=get_main_keyboard())
            else:
                logging.error(f"Не удалось переключить модель на: {model_name}")
                await callback.message.edit_text(
                    f"❌ <b>Ошибка переключения модели</b>\n\n"
                    f"Модель: <code>{model_name}</code>\n"
                    f"Не удалось переключить модель. Попробуйте еще раз.\n\n"
                    f"💡 <b>Возможные причины:</b>\n"
                    f"• Модель не найдена в списке\n"
                    f"• WebUI не отвечает\n"
                    f"• Недостаточно памяти",
                    parse_mode="HTML"
                )
                # Отправляем новое сообщение с главной клавиатурой
                await callback.message.answer("Выберите действие:", reply_markup=get_main_keyboard())
                
        except Exception as e:
            logging.error(f"Исключение при переключении модели {model_name}: {e}")
            await callback.message.edit_text(
                f"❌ <b>Ошибка при переключении модели</b>\n\n"
                f"Модель: <code>{model_name}</code>\n"
                f"Ошибка: <code>{str(e)}</code>\n\n"
                f"Попробуйте еще раз или обратитесь к администратору.",
                parse_mode="HTML"
            )
            # Отправляем новое сообщение с главной клавиатурой
            await callback.message.answer("Выберите действие:", reply_markup=get_main_keyboard())
    
    await callback.answer()

@dp.callback_query(F.data.startswith("cancel_"))
async def cancel_generation(callback: types.CallbackQuery, state: FSMContext):
    """Отмена генерации"""
    log_user_callback(callback)
    if callback.data:
        task_id = callback.data.replace("cancel_", "")
        
        if task_id == "simple":
            # Отмена простой генерации
            await state.clear()
            if callback.message:
                await callback.message.edit_text("❌ Простая генерация отменена")
                await callback.message.answer("Выберите действие:", reply_markup=get_main_keyboard())
        else:
            # Отмена задачи в очереди
            if queue_manager.cancel_task(task_id):
                if callback.message:
                    await callback.message.edit_text("❌ Генерация отменена")
            else:
                if callback.message:
                    await callback.message.edit_text("❌ Не удалось отменить задачу")
    
    await callback.answer()

# Обработка текстовых сообщений
@dp.message(GenerationStates.waiting_for_prompt)
async def handle_prompt(message: types.Message, state: FSMContext):
    """Обработка промпта для генерации"""
    log_user_message(message)
    if not message.text:
        await message.answer("❌ Промпт не может быть пустым.", reply_markup=get_main_keyboard())
        await state.clear()
        return
    
    prompt = message.text
    
    # Сохраняем промпт
    await state.update_data(prompt=prompt)
    
    # Проверяем доступность SD
    if not sd_client.is_available():
        await message.answer("❌ Stable Diffusion WebUI недоступен. Проверьте, что он запущен.", reply_markup=get_main_keyboard())
        await state.clear()
        return
    
    try:
        # Добавляем задачу в очередь
        task = queue_manager.add_task(message.from_user.id, prompt)
        queue_position = queue_manager.get_queue_position(task.id)
        
        # Отправляем сообщение о добавлении в очередь
        status_msg = await message.answer(
            f"📋 <b>Задача добавлена в очередь</b>\n\n"
            f"📝 Промпт: <code>{prompt}</code>\n\n"
            f"📊 Позиция в очереди: {queue_position}\n"
            f"⏳ Ожидание обработки...",
            reply_markup=get_generation_keyboard(task.id),
            parse_mode="HTML"
        )
        
        # Если это первая задача, начинаем обработку
        if queue_position == 1:
            # Запускаем обработку очереди в фоне
            asyncio.create_task(process_generation_queue())
        
        # Обновляем сообщение каждые 2 секунды
        while task.status == GenerationStatus.QUEUED:
            await asyncio.sleep(2)
            queue_position = queue_manager.get_queue_position(task.id)
            if queue_position > 0:
                await status_msg.edit_text(
                    f"📋 <b>Ожидание в очереди</b>\n\n"
                    f"📝 Промпт: <code>{prompt}</code>\n\n"
                    f"📊 Позиция в очереди: {queue_position}\n"
                    f"⏳ Ожидание обработки...",
                    reply_markup=get_generation_keyboard(task.id),
                    parse_mode="HTML"
                )
            else:
                break
        
        # Обновляем прогресс во время обработки
        while task.status == GenerationStatus.PROCESSING:
            await update_progress_message(task, status_msg)
            await asyncio.sleep(1)
            
        # Удаляем сообщение о прогрессе
        await status_msg.delete()
        
    except Exception as e:
        await message.answer(f"❌ Произошла ошибка: {str(e)}", reply_markup=get_main_keyboard())
    
    finally:
        await state.clear()

# Обработка простой пошаговой генерации
@dp.message(GenerationStates.waiting_for_animal_type)
async def handle_animal_type(message: types.Message, state: FSMContext):
    """Обработка вида животного"""
    log_user_message(message)
    if not message.text:
        await message.answer("❌ Пожалуйста, введите вид животного на английском языке.")
        return
    
    animal_type = message.text.strip()
    await state.update_data(animal_type=animal_type)
    await state.set_state(GenerationStates.waiting_for_gender)
    
    await message.answer(
        f"👤 <b>Шаг 2/10:</b> Пол персонажа\n\n"
        f"Выбрано: <b>{animal_type}</b>\n\n"
        f"💡 <b>Примеры:</b> male, female, boy, girl\n\n"
        f"⚠️ <b>Важно:</b> Пишите на английском языке!",
        reply_markup=get_simple_generation_keyboard(),
        parse_mode="HTML"
    )

@dp.message(GenerationStates.waiting_for_gender)
async def handle_gender(message: types.Message, state: FSMContext):
    """Обработка пола персонажа"""
    log_user_message(message)
    if not message.text:
        await message.answer("❌ Пожалуйста, введите пол персонажа на английском языке.")
        return
    
    gender = message.text.strip()
    data = await state.get_data()
    animal_type = data.get('animal_type', '')
    
    await state.update_data(gender=gender)
    await state.set_state(GenerationStates.waiting_for_fur_color)
    
    await message.answer(
        f"🎨 <b>Шаг 3/10:</b> Цвет меха\n\n"
        f"Выбрано: <b>{gender} {animal_type}</b>\n\n"
        f"💡 <b>Примеры:</b> red, brown, white, black, gray, orange, golden, silver\n\n"
        f"⚠️ <b>Важно:</b> Пишите на английском языке!",
        reply_markup=get_simple_generation_keyboard(),
        parse_mode="HTML"
    )

@dp.message(GenerationStates.waiting_for_fur_color)
async def handle_fur_color(message: types.Message, state: FSMContext):
    """Обработка цвета меха"""
    log_user_message(message)
    if not message.text:
        await message.answer("❌ Пожалуйста, введите цвет меха на английском языке.")
        return
    
    fur_color = message.text.strip()
    data = await state.get_data()
    animal_type = data.get('animal_type', '')
    gender = data.get('gender', '')
    
    await state.update_data(fur_color=fur_color)
    await state.set_state(GenerationStates.waiting_for_clothing)
    
    await message.answer(
        f"👕 <b>Шаг 4/10:</b> Одежда\n\n"
        f"Выбрано: <b>{gender} {fur_color} {animal_type}</b>\n\n"
        f"💡 <b>Примеры:</b> t-shirt, jacket, dress, armor, hoodie, sweater, bikini, swimsuit\n\n"
        f"⚠️ <b>Важно:</b> Пишите на английском языке!",
        reply_markup=get_simple_generation_keyboard(),
        parse_mode="HTML"
    )

@dp.message(GenerationStates.waiting_for_clothing)
async def handle_clothing(message: types.Message, state: FSMContext):
    """Обработка одежды"""
    log_user_message(message)
    if not message.text:
        await message.answer("❌ Пожалуйста, введите одежду на английском языке.")
        return
    
    clothing = message.text.strip()
    data = await state.get_data()
    animal_type = data.get('animal_type', '')
    gender = data.get('gender', '')
    fur_color = data.get('fur_color', '')
    
    await state.update_data(clothing=clothing)
    await state.set_state(GenerationStates.waiting_for_pose)
    
    await message.answer(
        f"🎭 <b>Шаг 5/10:</b> Поза\n\n"
        f"Выбрано: <b>{gender} {fur_color} {animal_type}</b> в <b>{clothing}</b>\n\n"
        f"💡 <b>Примеры:</b> standing, sitting, running, jumping, dancing, swimming, playing\n\n"
        f"⚠️ <b>Важно:</b> Пишите на английском языке!",
        reply_markup=get_simple_generation_keyboard(),
        parse_mode="HTML"
    )

@dp.message(GenerationStates.waiting_for_pose)
async def handle_pose(message: types.Message, state: FSMContext):
    """Обработка позы персонажа"""
    log_user_message(message)
    if not message.text:
        await message.answer("❌ Пожалуйста, введите позу на английском языке.")
        return
    
    pose = message.text.strip()
    data = await state.get_data()
    animal_type = data.get('animal_type', '')
    gender = data.get('gender', '')
    fur_color = data.get('fur_color', '')
    clothing = data.get('clothing', '')
    
    await state.update_data(pose=pose)
    await state.set_state(GenerationStates.waiting_for_expression)
    
    await message.answer(
        f"😊 <b>Шаг 6/10:</b> Выражение лица\n\n"
        f"Выбрано: <b>{gender} {fur_color} {animal_type}</b> в <b>{clothing}</b>, <b>{pose}</b>\n\n"
        f"💡 <b>Примеры:</b> happy, sad, angry, surprised, cute, confident, shy, excited\n\n"
        f"⚠️ <b>Важно:</b> Пишите на английском языке!",
        reply_markup=get_simple_generation_keyboard(),
        parse_mode="HTML"
    )

@dp.message(GenerationStates.waiting_for_expression)
async def handle_expression(message: types.Message, state: FSMContext):
    """Обработка выражения лица"""
    log_user_message(message)
    if not message.text:
        await message.answer("❌ Пожалуйста, введите выражение лица на английском языке.")
        return
    
    expression = message.text.strip()
    data = await state.get_data()
    animal_type = data.get('animal_type', '')
    gender = data.get('gender', '')
    fur_color = data.get('fur_color', '')
    clothing = data.get('clothing', '')
    pose = data.get('pose', '')
    
    await state.update_data(expression=expression)
    await state.set_state(GenerationStates.waiting_for_location)
    
    await message.answer(
        f"🌍 <b>Шаг 7/10:</b> Местность\n\n"
        f"Выбрано: <b>{gender} {fur_color} {animal_type}</b> в <b>{clothing}</b>, <b>{pose}</b>, <b>{expression}</b>\n\n"
        f"💡 <b>Примеры:</b> forest, city, beach, mountains, swimming pool, park, garden, fantasy world\n\n"
        f"⚠️ <b>Важно:</b> Пишите на английском языке!",
        reply_markup=get_simple_generation_keyboard(),
        parse_mode="HTML"
    )

@dp.message(GenerationStates.waiting_for_location)
async def handle_location(message: types.Message, state: FSMContext):
    """Обработка местности"""
    log_user_message(message)
    if not message.text:
        await message.answer("❌ Пожалуйста, введите местность на английском языке.")
        return
    
    location = message.text.strip()
    data = await state.get_data()
    animal_type = data.get('animal_type', '')
    gender = data.get('gender', '')
    fur_color = data.get('fur_color', '')
    clothing = data.get('clothing', '')
    pose = data.get('pose', '')
    expression = data.get('expression', '')
    
    await state.update_data(location=location)
    await state.set_state(GenerationStates.waiting_for_activity)
    
    await message.answer(
        f"🎯 <b>Шаг 8/10:</b> Действие/Активность\n\n"
        f"Выбрано: <b>{gender} {fur_color} {animal_type}</b> в <b>{clothing}</b>, <b>{pose}</b>, <b>{expression}</b>, в <b>{location}</b>\n\n"
        f"💡 <b>Примеры:</b> playing, swimming, relaxing, working, exploring, celebrating, training\n\n"
        f"⚠️ <b>Важно:</b> Пишите на английском языке!",
        reply_markup=get_simple_generation_keyboard(),
        parse_mode="HTML"
    )

@dp.message(GenerationStates.waiting_for_activity)
async def handle_activity(message: types.Message, state: FSMContext):
    """Обработка действия/активности"""
    log_user_message(message)
    if not message.text:
        await message.answer("❌ Пожалуйста, введите действие на английском языке.")
        return
    
    activity = message.text.strip()
    data = await state.get_data()
    animal_type = data.get('animal_type', '')
    gender = data.get('gender', '')
    fur_color = data.get('fur_color', '')
    clothing = data.get('clothing', '')
    pose = data.get('pose', '')
    expression = data.get('expression', '')
    location = data.get('location', '')
    
    await state.update_data(activity=activity)
    await state.set_state(GenerationStates.waiting_for_priority)
    
    await message.answer(
        f"⭐ <b>Шаг 9/10:</b> Приоритет отрисовки\n\n"
        f"Выбрано: <b>{gender} {fur_color} {animal_type}</b> в <b>{clothing}</b>, <b>{pose}</b>, <b>{expression}</b>, в <b>{location}</b>, <b>{activity}</b>\n\n"
        f"💡 <b>Выберите приоритет (0-5):</b>\n"
        f"• 0 - Обычное качество (вес: 1.0)\n"
        f"• 1 - Хорошее качество (вес: 1.1)\n"
        f"• 2 - Отличное качество (вес: 1.2)\n"
        f"• 3 - Превосходное качество (вес: 1.3)\n"
        f"• 4 - Максимальное качество (вес: 1.4)\n"
        f"• 5 - Ультра качество (вес: 1.5)\n\n"
        f"⚠️ <b>Важно:</b> Введите число от 0 до 5!",
        reply_markup=get_simple_generation_keyboard(),
        parse_mode="HTML"
    )

@dp.message(GenerationStates.waiting_for_priority)
async def handle_priority(message: types.Message, state: FSMContext):
    """Обработка приоритета отрисовки и завершение генерации"""
    log_user_message(message)
    if not message.text:
        await message.answer("❌ Пожалуйста, введите приоритет отрисовки (0-5).")
        return
    
    try:
        priority = int(message.text.strip())
        if priority < 0 or priority > 5:
            await message.answer("❌ Приоритет должен быть от 0 до 5. Попробуйте снова.")
            return
    except ValueError:
        await message.answer("❌ Пожалуйста, введите только цифру от 0 до 5.")
        return
    
    data = await state.get_data()
    
    # Собираем все данные
    animal_type = data.get('animal_type', '')
    gender = data.get('gender', '')
    fur_color = data.get('fur_color', '')
    clothing = data.get('clothing', '')
    pose = data.get('pose', '')
    expression = data.get('expression', '')
    location = data.get('location', '')
    activity = data.get('activity', '')
    
    # Конвертируем приоритет в вес
    weight = 1.0 + (priority * 0.1)
    
    # Формируем детальный промпт с весами
    prompt_parts = [
        f"({animal_type}:{weight:.1f})",
        f"({gender}:{weight:.1f})",
        f"({fur_color} fur:{weight:.1f})",
        f"({clothing}:{weight:.1f})",
        f"({location}:{weight:.1f})",
        f"({activity}:{weight:.1f})",
        f"({expression} expression:{weight:.1f})",
        f"({pose} pose:{weight:.1f})",
        f"detailed, high quality, digital art"
    ]
    
    prompt = ", ".join(prompt_parts)
    
    # Проверяем доступность SD
    if not sd_client.is_available():
        await message.answer("❌ Stable Diffusion WebUI недоступен. Проверьте, что он запущен.", reply_markup=get_main_keyboard())
        await state.clear()
        return
    
    try:
        # Добавляем задачу в очередь
        task = queue_manager.add_task(message.from_user.id, prompt)
        queue_position = queue_manager.get_queue_position(task.id)
        
        # Отправляем сообщение о добавлении в очередь
        status_msg = await message.answer(
            f"📋 <b>Задача добавлена в очередь</b>\n\n"
            f"🎨 <b>Созданный персонаж:</b>\n"
            f"• Вид: {animal_type}\n"
            f"• Пол: {gender}\n"
            f"• Цвет: {fur_color}\n"
            f"• Одежда: {clothing}\n"
            f"• Поза: {pose}\n"
            f"• Выражение: {expression}\n"
            f"• Местность: {location}\n"
            f"• Действие: {activity}\n"
            f"• Приоритет: {priority} ({weight:.1f})\n\n"
            f"📝 <b>Промпт:</b> <code>{prompt}</code>\n\n"
            f"📊 Позиция в очереди: {queue_position}\n"
            f"⏳ Ожидание обработки...",
            reply_markup=get_generation_keyboard(task.id),
            parse_mode="HTML"
        )
        
        # Запускаем обработку очереди в фоне (если еще не запущена)
        asyncio.create_task(process_generation_queue())
        
        # Запускаем мониторинг прогресса в фоне
        asyncio.create_task(monitor_task_progress(task, status_msg))
        
    except Exception as e:
        await message.answer(f"❌ Произошла ошибка: {str(e)}", reply_markup=get_main_keyboard())
    
    finally:
        await state.clear()

# Обработка продвинутых состояний
@dp.message(AdvancedGenerationStates.waiting_for_prompt)
async def handle_advanced_prompt(message: types.Message, state: FSMContext):
    """Обработка основного промпта для продвинутой генерации"""
    log_user_message(message)
    await advanced_features.handle_advanced_prompt(message, state)

@dp.message(AdvancedGenerationStates.waiting_for_negative_prompt)
async def handle_negative_prompt(message: types.Message, state: FSMContext):
    """Обработка негативного промпта"""
    log_user_message(message)
    await advanced_features.handle_negative_prompt(message, state)

@dp.message(AdvancedGenerationStates.waiting_for_steps)
async def handle_steps(message: types.Message, state: FSMContext):
    """Обработка количества шагов"""
    log_user_message(message)
    await advanced_features.handle_steps(message, state)

@dp.message(AdvancedGenerationStates.waiting_for_cfg_scale)
async def handle_cfg_scale(message: types.Message, state: FSMContext):
    """Обработка CFG Scale"""
    log_user_message(message)
    await advanced_features.handle_cfg_scale(message, state)

@dp.message(AdvancedGenerationStates.waiting_for_size)
async def handle_size(message: types.Message, state: FSMContext):
    """Обработка размера изображения"""
    log_user_message(message)
    await advanced_features.handle_size(message, state)

# Обработка обычных текстовых сообщений (автогенерация)
@dp.message(F.text)
async def handle_text_message(message: types.Message):
    """Обработка обычных текстовых сообщений как промптов"""
    log_user_message(message)
    prompt = message.text
    
    # Проверяем доступность SD
    if not sd_client.is_available():
        await message.answer("❌ Stable Diffusion WebUI недоступен. Проверьте, что он запущен.", reply_markup=get_main_keyboard())
        return
    
    try:
        # Добавляем задачу в очередь
        task = queue_manager.add_task(message.from_user.id, prompt)
        queue_position = queue_manager.get_queue_position(task.id)
        
        # Отправляем сообщение о добавлении в очередь
        status_msg = await message.answer(
            f"📋 <b>Задача добавлена в очередь</b>\n\n"
            f"📝 Промпт: <code>{prompt}</code>\n\n"
            f"📊 Позиция в очереди: {queue_position}\n"
            f"⏳ Ожидание обработки...",
            reply_markup=get_generation_keyboard(task.id),
            parse_mode="HTML"
        )
        
        # Запускаем обработку очереди в фоне (если еще не запущена)
        asyncio.create_task(process_generation_queue())
        
        # Запускаем мониторинг прогресса в фоне
        asyncio.create_task(monitor_task_progress(task, status_msg))
        
    except Exception as e:
        await message.answer(f"❌ Произошла ошибка: {str(e)}", reply_markup=get_main_keyboard())

# Обработка ошибок
@dp.errors()
async def errors_handler(update: types.Update, exception: Exception):
    """Обработчик ошибок"""
    logging.error(f"Ошибка при обработке {update}: {exception}")
    return True

async def main():
    """Главная функция"""
    logging.info("🚀 Запуск расширенного бота с клавиатурой и очередью...")
    
    # Проверяем доступность SD WebUI
    if sd_client.is_available():
        logging.info("✅ Stable Diffusion WebUI доступен")
    else:
        logging.warning("⚠️ Stable Diffusion WebUI недоступен")
    
    # Запускаем обработку очереди в фоне
    asyncio.create_task(process_generation_queue())
    
    # Запускаем бота
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main()) 