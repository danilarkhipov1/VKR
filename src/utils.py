from PIL import Image
import io
from typing import Tuple


def load_and_preprocess_image(image_data: bytes) -> Image.Image:
    """
    Загрузка и предобработка изображения

    Args:
        image_data: Байтовые данные изображения

    Returns:
        PIL.Image: Обработанное изображение
    """
    # Открываем изображение
    image = Image.open(io.BytesIO(image_data))

    # Конвертируем в RGB если изображение в другом формате
    if image.mode != 'RGB':
        image = image.convert('RGB')

    # Изменяем размер до 224x224 (требование ViT)
    image = image.resize((224, 224))

    return image


def format_prediction_result(
        predicted_classes: list,
        confidence_scores: list
) -> dict:
    """
    Форматирование результатов предсказания

    Args:
        predicted_classes: Список предсказанных классов
        confidence_scores: Список значений уверенности

    Returns:
        dict: Отформатированный результат
    """
    predictions = [
        {
            "class": class_name,
            "confidence": float(confidence)
        }
        for class_name, confidence in zip(predicted_classes, confidence_scores)
    ]

    return {
        "predictions": predictions,
        "number_of_classes": len(predictions)
    }


def validate_image(image: Image.Image) -> Tuple[bool, str]:
    """
    Проверка валидности изображения

    Args:
        image: PIL Image объект

    Returns:
        Tuple[bool, str]: Результат проверки и сообщение об ошибке
    """
    if image.size[0] < 32 or image.size[1] < 32:
        return False, "Изображение слишком маленькое"

    if image.size[0] > 4096 or image.size[1] > 4096:
        return False, "Изображение слишком большое"

    return True, ""