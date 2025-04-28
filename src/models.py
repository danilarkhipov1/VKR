from transformers import ViTForImageClassification, ViTFeatureExtractor
import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple


class WasteClassifier:
    def __init__(self, model_path: str = None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')
        self.model = ViTForImageClassification.from_pretrained(
            'google/vit-base-patch16-224-in21k',
            num_labels=15,  # 15 классов мусорных баков
            problem_type='multi_label_classification'
        )

        if model_path:
            self.load_model(model_path)

        self.model.to(self.device)
        self.model.eval()

        # Названия классов (замените на ваши реальные классы)
        self.class_names = [
            "переполненные ТКО",
            "не переполненные контейнеры ТКО",
            "переполненная площадка КГО",
            "не переполненная площадка КГО",
            "исправное оборудование",
            "неисправное оборудование",
            "закрытые крышки",
            "незакрытые крышки",
            "контейнеры вне места накопления",
            "контейнеры на месте накопления",
            "контейнер не предусматривает крышку",
            "контейнер не полный, а мусор сложен рядом",
            "на фото нет контейнеров или площадки для КГО, а только мусор или ничего",
            "переполненный контейнер КГО",
            "не переполненный контейнер КГО"
        ]

    def load_model(self, model_path: str) -> None:
        """Загрузка весов модели"""
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))

    def predict(self, image) -> Tuple[List[str], List[float]]:
        """
        Предсказание классов для изображения

        Args:
            image: PIL Image объект

        Returns:
            Tuple[List[str], List[float]]: Список предсказанных классов и их вероятности
        """
        # Подготовка изображения
        inputs = self.feature_extractor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.sigmoid(outputs.logits)

        # Получаем вероятности для каждого класса
        probs = predictions[0].cpu().numpy()

        # Получаем индексы классов с вероятностью > 0.5
        predicted_classes = np.where(probs > 0.5)[0]

        # Если ни один класс не превысил порог, берем класс с максимальной вероятностью
        if len(predicted_classes) == 0:
            predicted_classes = [np.argmax(probs)]

        # Формируем результат
        predicted_labels = [self.class_names[idx] for idx in predicted_classes]
        predicted_probs = [float(probs[idx]) for idx in predicted_classes]

        return predicted_labels, predicted_probs