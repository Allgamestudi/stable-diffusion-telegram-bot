import asyncio
import time
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass
from enum import Enum

class GenerationStatus(Enum):
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class GenerationStage(Enum):
    INITIALIZING = "initializing"
    LOADING_MODEL = "loading_model"
    PROCESSING_PROMPT = "processing_prompt"
    GENERATING_IMAGE = "generating_image"
    ENCODING_RESULT = "encoding_result"
    FINALIZING = "finalizing"

@dataclass
class GenerationTask:
    id: str
    user_id: int
    prompt: str
    status: GenerationStatus
    stage: GenerationStage
    created_at: float
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    progress: float = 0.0
    result: Optional[Dict] = None
    error: Optional[str] = None
    parameters: Optional[Dict] = None

class QueueManager:
    def __init__(self):
        self.queue: List[GenerationTask] = []
        self.processing: Optional[GenerationTask] = None
        self.completed_tasks: List[GenerationTask] = []
        self.task_counter = 0
        self.max_queue_size = 50
        self.max_completed_tasks = 100
        
    def add_task(self, user_id: int, prompt: str, parameters: Optional[Dict] = None) -> GenerationTask:
        """Добавляет задачу в очередь"""
        if len(self.queue) >= self.max_queue_size:
            raise Exception("Очередь переполнена. Попробуйте позже.")
        
        self.task_counter += 1
        task = GenerationTask(
            id=f"task_{self.task_counter}_{int(time.time())}",
            user_id=user_id,
            prompt=prompt,
            status=GenerationStatus.QUEUED,
            stage=GenerationStage.INITIALIZING,
            created_at=time.time(),
            parameters=parameters or {}
        )
        
        self.queue.append(task)
        return task
    
    def get_queue_position(self, task_id: str) -> int:
        """Получает позицию задачи в очереди"""
        for i, task in enumerate(self.queue):
            if task.id == task_id:
                return i + 1
        return -1
    
    def get_queue_info(self) -> Dict:
        """Получает информацию о очереди"""
        return {
            "queue_length": len(self.queue),
            "processing": self.processing is not None,
            "total_tasks": self.task_counter,
            "completed_tasks": len(self.completed_tasks)
        }
    
    def start_processing(self) -> Optional[GenerationTask]:
        """Начинает обработку следующей задачи"""
        if self.processing is not None:
            return None
        
        if not self.queue:
            return None
        
        task = self.queue.pop(0)
        task.status = GenerationStatus.PROCESSING
        task.started_at = time.time()
        self.processing = task
        
        return task
    
    def update_task_progress(self, task_id: str, stage: GenerationStage, progress: float):
        """Обновляет прогресс задачи"""
        if self.processing and self.processing.id == task_id:
            self.processing.stage = stage
            self.processing.progress = progress
    
    def complete_task(self, task_id: str, result: Dict):
        """Завершает задачу успешно"""
        if self.processing and self.processing.id == task_id:
            self.processing.status = GenerationStatus.COMPLETED
            self.processing.completed_at = time.time()
            self.processing.result = result
            self.completed_tasks.append(self.processing)
            
            # Ограничиваем количество сохраненных задач
            if len(self.completed_tasks) > self.max_completed_tasks:
                self.completed_tasks.pop(0)
            
            self.processing = None
    
    def fail_task(self, task_id: str, error: str):
        """Завершает задачу с ошибкой"""
        if self.processing and self.processing.id == task_id:
            self.processing.status = GenerationStatus.FAILED
            self.processing.completed_at = time.time()
            self.processing.error = error
            self.processing = None
    
    def cancel_task(self, task_id: str) -> bool:
        """Отменяет задачу"""
        # Отменяем из очереди
        for i, task in enumerate(self.queue):
            if task.id == task_id:
                task.status = GenerationStatus.CANCELLED
                self.queue.pop(i)
                return True
        
        # Отменяем текущую задачу
        if self.processing and self.processing.id == task_id:
            self.processing.status = GenerationStatus.CANCELLED
            self.processing = None
            return True
        
        return False
    
    def get_user_tasks(self, user_id: int) -> List[GenerationTask]:
        """Получает задачи пользователя"""
        user_tasks = []
        
        # Задачи в очереди
        for task in self.queue:
            if task.user_id == user_id:
                user_tasks.append(task)
        
        # Текущая задача
        if self.processing and self.processing.user_id == user_id:
            user_tasks.append(self.processing)
        
        # Завершенные задачи
        for task in self.completed_tasks:
            if task.user_id == user_id:
                user_tasks.append(task)
        
        return user_tasks
    
    def cleanup_old_tasks(self, max_age_hours: int = 24):
        """Очищает старые завершенные задачи"""
        current_time = time.time()
        max_age_seconds = max_age_hours * 3600
        
        self.completed_tasks = [
            task for task in self.completed_tasks
            if task.completed_at and (current_time - task.completed_at) < max_age_seconds
        ]

# Глобальный экземпляр менеджера очереди
queue_manager = QueueManager() 